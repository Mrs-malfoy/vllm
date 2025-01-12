import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    _current_load_budget: ClassVar[int] = 512
    token_budget: int
    max_num_seqs: int
    load_budget: int = 512   # 假设最大并行度是10
    hybrid_batch_time_budget: float = 9999
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0
    _sum_load: float = 0.0


    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        #assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    # Optimization for fast-access to seq_group lists
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[ScheduledSequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]
    swapped_out: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
            swapped_out=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[ScheduledSequenceGroup]
    # Blocks to swap out during preemption. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    swapped_out: List[SequenceGroup]  # 添加这个属性
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            blocks_to_swap_out=[],  # 添加空的 blocks_to_swap_out
            swapped_out=[],  # 初始化
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


def seq_group_metadata_builder():
    return SequenceGroupMetadata(request_id="",
                                 is_prompt=False,
                                 seq_data={},
                                 sampling_params=None,
                                 block_tables={})


def scheduler_running_outputs_builder():
    return SchedulerRunningOutputs(decode_seq_groups=[],
                                   prefill_seq_groups=[],
                                   preempted=[],
                                   swapped_out=[],
                                   blocks_to_swap_out=[],
                                   blocks_to_copy=[],
                                   num_lookahead_slots=0,
                                   prefill_seq_groups_list=[],
                                   decode_seq_groups_list=[])


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(SequenceGroup("", [], -1),
                                  token_chunk_size=0)
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config
        
        self.max_wait_time = scheduler_config.max_wait_time

        # self.ttft_slo = self.scheduler_config.ttft_slo
        # self.tbt_slo = self.scheduler_config.tbt_slo
        # self.swap_overhead = 0.08  # 80ms的swap开销
        self.decode_overhead = 0.02  # 20ms的decode开销
        self.chunked_prefill_overhead = 0.06     # 60ms的chunked prefill开销
        self.prefill_rate = 0.08    # 预估prefill速率

        self.max_hybrid_batch_time = 99999
        self.max_hybrid_batch_bs = 4096
        self.safe_headroom = 0.1 # 安全余量
        self.min_hybrid_batch_bs = 256

        # 4090 LLaMa3 8B
        self.dcp_predict_bs_factor = 0.00011018 
        self.dcp_predict_token_factor = 0.00000147  
        self.swap_overhead_factor = 0.000243 #每个block的swap开销 单位秒
        self.load_factor = 0.1

        # A800*2 QWen 35B
        # self.dcp_predict_bs_factor = 0.00016848
        # self.dcp_predict_token_factor = 0.00000097
        # self.swap_overhead_factor = TBD #每个block的swap开销 单位秒

        version = "selfattn"
        if (self.scheduler_config.embedding_mode
                or self.cache_config.is_attention_free):
            version = "placeholder"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def _should_force_schedule(self, seq_group: SequenceGroup, alloc_status: AllocStatus) -> bool:
        """检查序列组是否需要强制调度
        
        Args:
            seq_group: 待检查的序列组
            alloc_status: 资源分配状态
            
        Returns:
            bool: 是否需要强制调度
        """
        # 只对LATER状态的序列进行强制调度检查
        if alloc_status != AllocStatus.LATER:
            return False
            
        current_time = time.time()
        wait_time = current_time - seq_group.arrival_time

        prompt_tokens_len = len(seq_group.seqs[0].prompt_token_ids)

        return wait_time + 0.46824 + (9.6e-5) * prompt_tokens_len > self.max_wait_time

    def _get_waiting_overtime(self, seq_group: SequenceGroup) -> float:
        """计算waiting队列中序列组的超时时长
        
        Args:
            seq_group: 待检查的序列组
            alloc_status: 资源分配状态
            
        Returns:
            float: 超出阈值的时长(秒),如果未超时则返回0
        """
        current_time = time.time()
        wait_time = current_time - seq_group.arrival_time
        
        # 计算预期等待时间
        prompt_tokens_len = len(seq_group.seqs[0].prompt_token_ids)
        expected_wait = 0.46824 + (9.6e-5) * prompt_tokens_len
        
        # 计算超出阈值的时长
        overtime = wait_time + expected_wait - self.max_wait_time
        
        return max(0.0, overtime)

    def _get_swapped_overtime(self, seq_group: SequenceGroup) -> float:
        """计算swap队列中序列组的超时时长
        
        Args:
            seq_group: 待检查的序列组
            
        Returns:
            float: 超出阈值的时长(秒),如果未超时则返回0
        """
        if seq_group.is_finished():
            return 0.0
        
        if not seq_group.metrics or not seq_group.metrics.first_scheduled_time:
            return 0.0
            
        current_time = time.time()
        # 计算剩余可播放时间
        remaining_audio_time = (
            seq_group.seqs[0].seq_duration - 
            (current_time - seq_group.metrics.first_scheduled_time)
        )
        
        # 如果剩余时间小于1秒则认为超时
        # 这里的1.0是一个阈值,可以根据需要调整
        overtime = 0.18 - remaining_audio_time
        
        return max(0.0, overtime)

    def _get_most_urgent_waiting_overtime(self) -> float:
        """获取waiting队列中最紧急的超时时长"""
        if not self.waiting:
            return 0.0
        return self._get_waiting_overtime(self.waiting[0])

    def _get_most_urgent_swapped_overtime(self) -> float:
        """获取swap队列中最紧急的超时时长"""
        if not self.swapped:
            return 0.0
        return max(self._get_swapped_overtime(seq_group) for seq_group in self.swapped)

    def _get_running_headroom(self, seq_group: SequenceGroup) -> float:
        """计算running请求的余量
        
        Args:
            seq_group: 需要计算余量的序列组
            
        Returns:
            float: 时间余量(秒)。正值表示有富余,负值表示已超时
        """
        current_time = time.time()
        
        # 判断是否在prefill阶段
        if seq_group.is_prefill():
            # prefill阶段的计算方式
            elapsed_time = current_time - seq_group.arrival_time
            remaining_tokens = seq_group.get_num_uncomputed_tokens()
            
            # 计算还需要多少次chunk操作
            chunk_size = self.scheduler_config.max_num_batched_tokens
            num_chunks = (remaining_tokens + chunk_size - 1) // chunk_size  # 向上取整
            
            # 计算预期剩余时间
            elapsed_time = current_time - seq_group.arrival_time
            expected_time = num_chunks * self.chunked_prefill_overhead
            headroom = seq_group.seqs[0].ttft_slo - (elapsed_time + expected_time)
            print(f"calc headroom for running prefill! seq.request_id:{seq_group.request_id}, seq.wait_time:{time.time() - seq_group.arrival_time}, headrrom:{headroom}, slotype:{seq_group.seqs[0].slo_class}, ttft_slo:{seq_group.seqs[0].ttft_slo}")
            # print(f"seq.request_id:{seq_group.request_id}, seq.wait_time:{time.time() - seq_group.arrival_time}, headrrom:{headroom}")
        
            
        else:
            # decode阶段的计算方式
            if seq_group.metrics.first_token_time is None:
                return 0
            elapsed_time = current_time - seq_group.metrics.first_token_time
            num_tokens = seq_group.seqs[0].get_output_len()
            
            expected_time = num_tokens * seq_group.seqs[0].tbt_slo
            headroom = expected_time - (elapsed_time + self.decode_overhead)
        
        return headroom
    
    def _get_swapped_headroom(self, seq_group: SequenceGroup) -> float:
        """计算swapped请求的余量"""
        running_headroom = self._get_running_headroom(seq_group)
        # 需要额外考虑swap开销
        return running_headroom - 2 * self.swap_overhead_factor * seq_group.seqs[0].block_size
    
    def _get_most_urgent_swapped_headroom(self) -> float:
        """获取swap队列中最紧急的余量"""
        if not self.swapped:
            return float('inf')
        return min(self._get_swapped_headroom(seq_group) for seq_group in self.swapped)

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = \
            self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        # 24/11/30 feat: 在running队列里也遵循，按剩余可播放时间排序,优先抢占剩余时间长的序列
        self.running = deque(sorted(
            self.running,
            key=lambda x: (
                self._get_running_headroom(x) + (10000 if x.is_prefill() else 0)
            ),
            # reverse=True
        ))

        for seq in self.running:
            if(seq.is_prefill()):
                print(f"seq.request_id:{seq.request_id}, seq.wait_time:{time.time() - seq.arrival_time}, headrrom:{self._get_running_headroom(seq)}")
        
        # print(f"running before budget.num_batched_tokens:{budget.num_batched_tokens}")
        # print(f"running before budget.num_curr_seqs:{budget.num_curr_seqs}")
        # print(f"running: len(self.running):{len(self.running)}")
        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if self.use_async_output_proc and seq_group.seqs[0].get_len(
            ) > self.scheduler_config.max_model_len:
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                
                # budget._sum_load += self.chunked_prefill_overhead / self.tbt_slo
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        # print(f"prefill_seq_groups:{len(prefill_seq_groups)}")
        # print(f"decode_seq_groups:{len(decode_seq_groups)}")
        # print(f"running budget.num_batched_tokens:{budget.num_batched_tokens}")
        # print(f"running budget.num_curr_seqs:{budget.num_curr_seqs}")
        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # print("haha, please swap me!")
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_swap_out: List[Tuple[int, int]] = []  #从running队列抢占被换出
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []
 
        # print(f"Swapped:{self.swapped}")
        # print(f"running:{self.running}")
        self.swapped = deque(sorted(
            self.swapped,
            key=lambda x: (
                self._get_swapped_headroom(x)
            )
        ))
        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]
            # 24/12/17 feat: 添加一个对完成状态的判断
            if seq_group.is_finished():
                swapped_queue.popleft()
                continue
            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                logger.info("swapped LATER begin")
                # 如果剩余时间小于阈值,尝试强制恢复
                if self._get_swapped_headroom(seq_group) <= self.safe_headroom:  # 可配置的阈值
                    logger.info(
                        f"Attempting force schedule for sequence {seq_group.request_id} "
                    )
                    # print(f"force budget.num_batched_tokens:{budget.num_batched_tokens}")
                    # print(f"force budget.num_curr_seqs:{budget.num_curr_seqs}")
                    success, preempted_seqs = self._force_swap_in_by_preemption(
                        seq_group,
                        blocks_to_swap_in,
                        blocks_to_swap_out,
                        blocks_to_copy,
                        budget,
                        is_prefill,
                        enable_chunking,
                        prefill_seq_groups,
                        decode_seq_groups
                    )
                    # print("Q:are you success?")
                    # print(success)
                    
                    if success:
                        logger.info("force preempt success")
                        swapped_queue.popleft()
                        swapped_out.extend(preempted_seqs)  # 将被抢占的序列添加到swapped队列
                        continue
                    else:
                        logger.warning(
                            f"Failed to force schedule sequence {seq_group.request_id} "
                        )
                break    

            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
            # budget._sum_load += (self.chunked_prefill_overhead+self.swap_overhead) / self.tbt_slo

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            swapped_out=swapped_out,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled and \
                not self.scheduler_config.is_multi_step:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _get_priority(self,
                      seq_group: SequenceGroup) -> Tuple[Optional[int], float]:
        """ Get the priority of the sequence group.
        Highest preference to user-defined priority, followed by arrival time.
        Args:
            seq_group: The sequence group input.
        Returns:
            The priority of the sequence group.
        """
        return seq_group.priority, seq_group.arrival_time

    def _schedule_priority_preemption(
        self,
        budget: SchedulingBudget,
    ) -> int:
        """Sorts waiting and running queue. Also, force preempt requests
        from the running queue if their priority is lower.
        Priority-based preemption is used with the priority policy.
        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
        Returns:
            A count of priority-based preemptions.
        """

        waiting_queue = self.waiting

        running_queue = deque(sorted(self.running, key=self._get_priority))

        blocks_to_swap_out: List[Tuple[int, int]] = []
        force_preemption_count = 0

        if waiting_queue:
            seq_group = waiting_queue.popleft()
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      False, budget)

            #Only preempt if priority inversion exists
            while running_queue and self._get_priority(
                    running_queue[-1]) > self._get_priority(seq_group):
                #Only preempt if waiting sequence cannot be allocated
                can_allocate = self.block_manager.can_allocate(seq_group)
                if (num_new_tokens and can_allocate == AllocStatus.OK
                        and budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break

                #Adjust budget to remove the victim sequence group
                vseq_group = running_queue.pop()
                num_running_tokens = self._get_num_new_tokens(
                    vseq_group, SequenceStatus.RUNNING, False, budget)
                budget.subtract_num_batched_tokens(vseq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = vseq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(vseq_group.request_id,
                                         num_running_seqs)

                #Preempt out the victim sequence group
                self._preempt(vseq_group, blocks_to_swap_out,
                              PreemptionMode.RECOMPUTE)
                waiting_queue.appendleft(vseq_group)
                force_preemption_count += 1
            #Put the sequence back into the waiting queue
            waiting_queue.appendleft(seq_group)

        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))

        self.waiting = waiting_queue
        self.running = running_queue
        return force_preemption_count

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """


        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_swap_out: List[Tuple[int, int]] = []
        swapped_out: List[SequenceGroup] = []

        self.waiting = deque(sorted(
            self.waiting,
            key=lambda x: (
                self._get_running_headroom(x)
            )
        ))
        for seq in self.waiting:
            print(f"for waiting request: seq.request_id:{seq.request_id}, seq.wait_time:{time.time() - seq.arrival_time}, headrrom:{self._get_running_headroom(seq)}")
        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            print(f"waiting_queue:{len(waiting_queue)}")
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            if can_allocate == AllocStatus.LATER:
                logger.info("prefill LATER begin")
                # 只对LATER状态检查是否需要强制调度
                if self._get_running_headroom(seq_group) <= self.safe_headroom:
                    logger.info(
                        f"Attempting force schedule for sequence {seq_group.request_id} "
                    )

                    success, preempted = self._force_preempt_for_waiting_seq(
                    seq_group, blocks_to_swap_in, blocks_to_swap_out, enable_chunking, budget, num_new_tokens)
                    if success:
                        logger.info("force preempt success")
                        # 抢占成功,继续处理该序列
                        swapped_out.extend(preempted)
                        waiting_queue.popleft()
                        self._allocate_and_set_running(seq_group)
                        
                        seq_groups.append(
                            ScheduledSequenceGroup(
                                seq_group=seq_group,
                                token_chunk_size=num_new_tokens
                            )
                        )

                        continue
                    else:
                        logger.warning(
                            f"Failed to force schedule sequence {seq_group.request_id} "
                        )
                break

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)


        # if len(waiting_queue) == 0:
        #     logger.warning("this step has no request left~")
        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            blocks_to_swap_out=blocks_to_swap_out,
            swapped_out=swapped_out,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def _force_swap_in_by_preemption(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
        blocks_to_copy: List[Tuple[int, int]],
        budget: SchedulingBudget,
        is_prefill: bool,
        enable_chunking: bool,
        prefill_seq_groups: List[ScheduledSequenceGroup],
        decode_seq_groups: List[ScheduledSequenceGroup]
    ) -> Tuple[bool, List[SequenceGroup]]:
        """通过抢占running队列中的序列来强制恢复一个序列
        
        Args:
            seq_group: 需要恢复的序列组
            blocks_to_swap_in: 需要换入的块列表
            blocks_to_swap_out: 需要换出的块列表
            budget: 调度预算
            is_prefill: 是否是prefill阶段
            enable_chunking: 是否启用分块
            
        Returns:
            Tuple[bool, Optional[ScheduledSequenceGroup], List[SequenceGroup]]: 
                - 是否成功恢复
                - 如果成功,返回调度后的序列组
                - 被抢占的序列组列表
        """
        # print(self.running)
        # 按剩余语音时间降序排序running队列
        self.running = deque(sorted(
            self.running,
            key=lambda x: (
               self._get_swapped_headroom(x) + (-10000 if x.is_prefill() else 0)
            ),
            reverse=True  # 降序,剩余时间多的优先被抢占
        ))
        
        preempted_seqs = []  # 记录被抢占的序列
        running_seqs = deque(self.running)
        # print(f"running_seqs:{len(running_seqs)}")
        # 尝试抢占直到能恢复当前序列
        for victim in running_seqs:
            # print(f"victim:{self._get_swapped_headroom(victim)}")
            print(f"chunked_prefill_overhead:{self.chunked_prefill_overhead}")
            if self._get_swapped_headroom(victim) <= self.chunked_prefill_overhead:  # 如果当前余量已经不支持做chunked prefill,则停止抢占
                break

            # print(f"before budget.num_batched_tokens:{budget.num_batched_tokens}")
            # print(f"before budget.num_curr_seqs:{budget.num_curr_seqs}")

            num_running_tokens = self._get_num_new_tokens(
                victim, SequenceStatus.RUNNING, enable_chunking, budget)
            budget.subtract_num_batched_tokens(victim.request_id, num_running_tokens)
            num_running_seqs = victim.get_max_num_running_seqs()
            budget.subtract_num_seqs(victim.request_id,
                                        num_running_seqs)

            # print(f"victim num_running_tokens:{num_running_tokens}")
            # print(f"victim num_running_seqs:{num_running_seqs}")            
            # print(f"after budget.num_batched_tokens:{budget.num_batched_tokens}")
            # print(f"after budget.num_curr_seqs:{budget.num_curr_seqs}")            
                
            self._preempt(victim, blocks_to_swap_out)
            self.running.popleft()
            preempted_seqs.append(victim)  # 添加到被抢占列表
            
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            
            if alloc_status == AllocStatus.OK:
                # 检查是否满足调度预算
                num_new_seqs = seq_group.get_max_num_running_seqs()
                num_new_tokens = self._get_num_new_tokens(
                    seq_group,
                    SequenceStatus.SWAPPED,
                    enable_chunking, 
                    budget
                )
                # print(f"num_new_tokens:{num_new_tokens}")
                # print(f"num_new_seqs:{num_new_seqs}")
                # print(f"budget.num_batched_tokens:{budget.num_batched_tokens}")
                # print(f"budget.num_curr_seqs:{budget.num_curr_seqs}")


                if (num_new_tokens == 0 or 
                    not budget.can_schedule(
                        num_new_tokens=num_new_tokens,
                        num_new_seqs=num_new_seqs
                    )):
                    break
                    
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)

                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(seq_group,
                                            token_chunk_size=num_new_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group, token_chunk_size=1))
                
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                # self.running = running_seqs #确认成功以后再给self.running重新赋值
                # print(f"success budget.num_batched_tokens:{budget.num_batched_tokens}")
                # print(f"success budget.num_curr_seqs:{budget.num_curr_seqs}")
                logger.info(f"success preempt for swapped seq")
                return True, preempted_seqs
        print("force swap not ok")
        for seq in preempted_seqs:
            self._swap_in(seq, blocks_to_swap_in)
            num_tokens = self._get_num_new_tokens(
                seq, SequenceStatus.RUNNING, enable_chunking, budget)
            budget.add_num_batched_tokens(seq.request_id, num_tokens)
            budget.add_num_seqs(seq.request_id, seq.get_max_num_running_seqs())
        
        # print(f"failed budget.num_batched_tokens:{budget.num_batched_tokens}")
        # print(f"failed budget.num_curr_seqs:{budget.num_curr_seqs}")
        self.running.extendleft(preempted_seqs)
        return False, []

    def _force_preempt_for_waiting_seq(
        self, 
        waiting_seq: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
        enable_chunking: bool,
        budget: SchedulingBudget,
        num_new_tokens: int
    ) -> Tuple[bool, List[SequenceGroup]]:
        """为等待过久的序列强制执行抢占
        
        Args:
            waiting_seq: 需要强制调度的序列组
            blocks_to_swap_out: 需要交换出的块列表
            
        Returns:
            bool: 是否成功执行抢占
        """    
        # 按剩余可播放时间排序,优先抢占剩余时间长的序列
        self.running = deque(sorted(
            self.running,
            key=lambda x: (
                self._get_swapped_headroom(x) + (-1000 if x.is_prefill() else 0)
            ),
            reverse=True
        ))
        # print(f"running: {running_seqs}")
        running_seqs = deque(self.running)
        
        preempted_seqs = []
        # print(f"running_seqs:{len(running_seqs)}")
        for victim in running_seqs:
            # print(f"victim:{self._get_swapped_headroom(victim)}")
            # print(f"chunked_prefill_overhead:{self.chunked_prefill_overhead}")
            if self._get_swapped_headroom(victim) <= self.chunked_prefill_overhead:  # 如果当前余量已经不支持做chunked prefill,则停止抢占
                break

            # print(f"budget.num_batched_tokens:{budget.num_batched_tokens}")
            # print(f"budget.num_curr_seqs:{budget.num_curr_seqs}")
            
            num_running_tokens = self._get_num_new_tokens(
                victim, SequenceStatus.RUNNING, enable_chunking, budget)
            budget.subtract_num_batched_tokens(victim.request_id, num_running_tokens)
            num_running_seqs = victim.get_max_num_running_seqs()
            budget.subtract_num_seqs(victim.request_id,
                                        num_running_seqs)
            
            # print(f"victim num_running_tokens:{num_running_tokens}")
            # print(f"victim num_running_seqs:{num_running_seqs}")
                
            self._preempt(victim, blocks_to_swap_out)
            preempted_seqs.append(victim)
            self.running.popleft()
            
            # 检查当前资源是否足够
            if self.block_manager.can_allocate(waiting_seq) == AllocStatus.OK:
                # fix: 加入对num_new_seqs的计算
                num_new_seqs = waiting_seq.get_max_num_running_seqs()

                # print(f"num_new_tokens:{num_new_tokens}")
                # print(f"num_new_seqs:{num_new_seqs}")
                # print(f"budget.num_batched_tokens:{budget.num_batched_tokens}")
                # print(f"budget.num_curr_seqs:{budget.num_curr_seqs}")

                if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break
                                        # fix: 增加预算
                budget.add_num_batched_tokens(waiting_seq.request_id, num_new_tokens)
                budget.add_num_seqs(waiting_seq.request_id, num_new_seqs)

                # print(f"success budget.num_batched_tokens:{budget.num_batched_tokens}")
                # print(f"success budget.num_curr_seqs:{budget.num_curr_seqs}")
                logger.info(f"success preempt for waiting seq")
                return True, preempted_seqs  # 返回成功状态和被抢占的序列
                
        # 如果抢占所有序列后仍无法分配,则恢复抢占的序列
        # for seq in preempted_seqs:
        #     self.running.append(seq)
            # 注意:_preempt()已经处理了blocks的swap,
            # 所以这里不需要额外处理blocks的恢复
        print("force preempt not ok")
        for seq in preempted_seqs:
            self._swap_in(seq, blocks_to_swap_in)
            num_tokens = self._get_num_new_tokens(
                seq, SequenceStatus.RUNNING, enable_chunking, budget)
            budget.add_num_batched_tokens(seq.request_id, num_tokens)
            budget.add_num_seqs(seq.request_id, seq.get_max_num_running_seqs())

        # print(f"failed budget.num_batched_tokens:{budget.num_batched_tokens}")
        # print(f"failed budget.num_curr_seqs:{budget.num_curr_seqs}")

        self.running.extendleft(preempted_seqs)
        return False, []

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        # print(f"begin,self.running:{len(self.running)}")
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()
        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped or (self.waiting and 
            self._get_most_urgent_waiting_overtime() > self._get_most_urgent_swapped_overtime()):
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        if len(prefills.seq_groups
               ) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            # print(f"running:{self.running}")
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)
            # print("_shcedule_running end")
            # print(f"running:{self.running}")
            # print(f"swapped:{self.swapped}")
            # print(f"waiting:{self.waiting}")
            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            # print(len(running_scheduled.preempted))
            # print(len(running_scheduled.swapped_out))
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                # print("we are going to use _schedule_swapped")
                self.running.extend(running_scheduled.decode_seq_groups_list)   #把running队列取出来方便swap运算
                # print(f"Swapped:{self.swapped}")
                # print(f"running:{self.running}")
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])
        
        # 恢复running队列
        if len(prefills.seq_groups) != 0 or len(running_scheduled.preempted) + len(
            running_scheduled.swapped_out) != 0:
            self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        all_swapped_out = running_scheduled.swapped_out + prefills.swapped_out + swapped_in.swapped_out
        # print(running_scheduled.swapped_out)
        # print(prefills.swapped_out)
        # print(swapped_in.swapped_out)
        self.swapped.extend(all_swapped_out)
    
        # Update swapped requests.
        #self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(all_swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
            scheduled_seq_groups.extend(swapped_in.decode_seq_groups)
        else:
            scheduled_seq_groups = [
                ScheduledSequenceGroup(seq_group=running_group,
                                        token_chunk_size=1) 
                for running_group in self.running
                ]
            # scheduled_seq_groups = []
            # scheduled_seq_groups.append(ScheduledSequenceGroup(seq_group=self.running[len(self.running)-1], token_chunk_size=1))
            # scheduled_seq_groups = []
            # for running_group in self.running:
            #     try:
            #         scheduled_seq_groups.append(ScheduledSequenceGroup(seq_group=running_group, token_chunk_size=1))
            #     except Exception as e:
            #         print(f"Error creating ScheduledSequenceGroup for {running_group}: {e}")

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        all_blocks_to_swap_out = (running_scheduled.blocks_to_swap_out + 
                             prefills.blocks_to_swap_out + swapped_in.blocks_to_swap_out)
        #all_swapped_out = running_scheduled.swapped_out + prefills.swapped_out
        
        # print(f"running:{len(self.running)}")
        # print(f"swapped:{len(self.swapped)}")
        # print(f"waiting:{len(self.waiting)}")
        # print(f"scheduled_seq_groups:{len(scheduled_seq_groups)}")
        # print(f"num_prefill_groups:{(num_prefill_groups)}")
        # print(f"num_batched_tokens:{(budget.num_batched_tokens)}")
        # print(f"ignored_seq_groups:{len(ignored_seq_groups)}")
        # print(f"preempted:{(preempted)}")
        # print(f"swapped_in.decode_seq_groups:{swapped_in.decode_seq_groups}")
        # print(f"running_scheduled.decode_seq_groups:{len(running_scheduled.decode_seq_groups)}")


        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=all_blocks_to_swap_out,  # 使用合并后的列表
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            load_budget=SchedulingBudget._current_load_budget,
        )
        # budget._sum_load += self.chunked_prefill_overhead / self.tbt_slo * len(self.running)
        # budget._sum_load += (self.swap_overhead + self.chunked_prefill_overhead) / self.tbt_slo * len(self.swapped)
        # print(f"budget._sum_load: {budget._sum_load}")
        min_headroom = 99999
        for seq_group in self.running:
            budget._sum_load += self.chunked_prefill_overhead / seq_group.seqs[0].tbt_slo
            min_headroom = min(min_headroom, self._get_running_headroom(seq_group))

        
        for seq_group in self.swapped:
            budget._sum_load += self.chunked_prefill_overhead / seq_group.seqs[0].tbt_slo
            min_headroom = min(min_headroom, self._get_swapped_headroom(seq_group))

        budget.hybrid_batch_time_budget = min_headroom # 不回滚_get_running_headroom里面计算过的decode时间，直接作为安全时间

        for seq_group in self.running:
            budget.hybrid_batch_time_budget -= seq_group.seqs[0].data.get_num_computed_tokens() * self.dcp_predict_token_factor + self.dcp_predict_bs_factor
        logger.info(f"budget.hybrid_batch_time_budget remains:{budget.hybrid_batch_time_budget}")
        dcp_cp_bs = int(budget.hybrid_batch_time_budget / (self.dcp_predict_token_factor + self.dcp_predict_bs_factor))
        dcp_hybrid_bs = dcp_cp_bs + len(self.running)
        dcp_hybrid_bs = dcp_hybrid_bs // 128 * 128
        dcp_hybrid_bs = max(self.min_hybrid_batch_bs, dcp_hybrid_bs)
        dcp_hybrid_bs = min(self.max_hybrid_batch_bs, dcp_hybrid_bs)
        self.scheduler_config.max_num_batched_tokens = dcp_hybrid_bs
        
        budget.token_budget = self.scheduler_config.max_num_batched_tokens
        can_schedule_more = True
        if len(self.running) > 0:
            budget._sum_load *= 1 + len(self.swapped)/(len(self.swapped) + len(self.running)) * 0.1   # 这个0.1为可修改参数 

            can_schedule_more = ((len(self.swapped) / len(self.running)) <= self.load_factor)

        logger.info(f"min_head_room:{min_headroom}, dcp_hybrid_bs:{self.scheduler_config.max_num_batched_tokens}, can_schedule_more:{can_schedule_more}")

        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)
        # print(f"running: {len(self.running)}")
        flag = False
        if len(self.running):
            flag = True
        # 提前恢复running队列，方便后续抢占
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        # print(f"running after extend: {len(self.running)}")

        # print(f"running after budget.num_batched_tokens:{budget.num_batched_tokens}")
        # print(f"running after budget.num_curr_seqs:{budget.num_curr_seqs}")
        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.

        if not flag:
            # 如果发生了抢占，则需要更新load_budget
            # 在这里更新是怕遇到那种几个cp把budget占完了的情况
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) != 0:
                budget.load_budget = len(running_scheduled.prefill_seq_groups) + len(running_scheduled.decode_seq_groups)
                SchedulingBudget._current_load_budget = budget.load_budget

            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0 or self._get_most_urgent_swapped_headroom() <= self.safe_headroom:
                swapped_in = self._schedule_swapped(budget, curr_loras)
                # print("swap finished")  # 注释

            # if budget._sum_load < budget.load_budget:
            if can_schedule_more:
                # Schedule new prefills.
                prefills = self._schedule_prefills(budget,
                                                curr_loras,
                                                enable_chunking=True)
            # print("prefill finished")  # 注释

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        # By default, vLLM scheduler prioritizes prefills.
        # Once chunked prefill is enabled,
        # the policy is changed to prioritize decode requests.
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])

        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out+prefills.swapped_out+swapped_in.swapped_out)

        filtered_decode_groups = [
            group for group in running_scheduled.decode_seq_groups
            if not any(seq.status == SequenceStatus.SWAPPED 
                    for seq in group.seq_group.get_seqs())
        ]
        
        filtered_prefill_groups = [
            group for group in running_scheduled.prefill_seq_groups
            if not any(seq.status == SequenceStatus.SWAPPED 
                    for seq in group.seq_group.get_seqs())
        ]
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  filtered_prefill_groups +
                                  swapped_in.prefill_seq_groups +
                                  filtered_decode_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(filtered_prefill_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=(running_scheduled.blocks_to_swap_out +
                                 prefills.blocks_to_swap_out +
                                 swapped_in.blocks_to_swap_out),
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup,
                          enable_chunking: bool) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        is_prefill = seq_group.is_prefill()
        num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        if is_prefill and num_lookahead_slots > 0:
            # Appending prefill slots only happens multi-step and
            # chunked-prefill are enabled together.
            assert self.scheduler_config.is_multi_step and enable_chunking

        return self.block_manager.can_append_slots(
            seq_group=seq_group, num_lookahead_slots=num_lookahead_slots)

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        # async_output_proc is allowed only when we have a single sequence
        # in the sequence group
        no_single_seq = seq_group.sampling_params is None or (
            seq_group.sampling_params.n == 1)
        return no_single_seq

    def schedule(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=seq_group.multi_modal_data
                    if scheduler_outputs.num_prefill_groups > 0 else None,
                    mm_processor_kwargs=seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (seq_group_metadata_list, scheduler_outputs,
                allow_async_output_proc)

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            # Free cross-attention block table, if it exists
            self._free_seq_group_cross_attn_blocks(seq_group)

            # Add the finished requests to the finished requests list.
            # This list will be used to update the Mamba cache in the
            # next step.
            self._finished_requests_ids.append(seq_group.request_id)

        # Free finished seqs
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        self.running = remaining

        # Handle async stopped sequence groups
        # (ones that reached max model len)
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)

                # Free finished seqs
                self._free_finished_seqs(seq_group)

            self._async_stopped.clear()

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(self,
                      seq_group: SequenceGroup,
                      blocks_to_copy: List[Tuple[int, int]],
                      enable_chunking: bool = False) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking)

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        # 如果明确指定了 preemption_mode，就使用指定的模式
        # if preemption_mode is not None:
        #     pass  # 保持传入的 preemption_mode 不变

        if self.user_specified_preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.user_specified_preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time

        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool,
                                 enable_chunking: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                # num_lookahead_slots was introduced in the context of decodes,
                # in Speculative Decoding.
                # When the num_scheduler_steps is 8, say, then the
                # num_lookahead_slots is 7. Meaning, we are doing a 1-step of
                # decode anyways and we wish to do 7 more.
                #
                # "lookaheads" for prefills, is introduced in support for
                # Chunked-Prefill in Multi-Step.
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns 0 if the new token cannot be computed due to token budget.
        """
        num_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()
        
        if num_new_tokens == 0:
            print(seq_group.get_seqs())
        assert num_new_tokens > 0
        # Chunk if a running request cannot fit in the given budget.
        # If number of seq > 1, it means it is doing beam search
        # in a decode phase. Do not chunk.
        if enable_chunking and len(seqs) == 1:
            remaining_token_budget = budget.remaining_token_budget()
            if self.scheduler_config.is_multi_step:
                # The current multi-step + chunked prefill capability does
                # not actually support chunking prompts.
                #
                # Therefore, `num_new_tokens` is computed in the same fashion
                # for both multi-step+chunked-prefill &
                # multi-step+chunked-prefill+APC
                #
                # Prompts with more tokens than the current remaining budget
                # are postponed to future scheduler steps
                if num_new_tokens > self._get_prompt_limit(seq_group):
                    # If the seq_group is in prompt-stage, pass the
                    # num_new_tokens as-is so the caller can ignore
                    # the sequence.
                    pass
                else:
                    num_new_tokens = 0 \
                        if num_new_tokens > remaining_token_budget \
                        else num_new_tokens
            elif self.cache_config.enable_prefix_caching:
                # When prefix caching is enabled, we always allocate
                # the number of new tokens that is dividable by the block
                # size to avoid partial block matching.
                block_size = self.cache_config.block_size
                remainder = budget.token_budget % block_size
                if remainder != 0:
                    raise ValueError("When enabling chunked prefill and "
                                     "prefix caching, max_num_batched_tokens "
                                     "(chunk size) must be dividable by "
                                     "block size, but got chunk_size "
                                     f"({budget.token_budget}) % block_size "
                                     f"({block_size}) = {remainder}")
                if remaining_token_budget < num_new_tokens:
                    num_new_tokens = (remaining_token_budget //
                                      block_size) * block_size
            else:
                num_new_tokens = min(num_new_tokens, remaining_token_budget)
        return num_new_tokens
