from abc import ABC, abstractmethod
from collections import deque
import time
from typing import Deque, List, Tuple, Set, Optional
from venv import logger

from vllm.core.interfaces import AllocStatus
from vllm.sequence import SequenceGroup, SequenceStatus
from typing import TYPE_CHECKING, List, Set, Optional

if TYPE_CHECKING:
    from vllm.core.scheduler import (
        Scheduler,
        SchedulerOutputs,
        SchedulerSwappedInOutputs,
        SchedulerPrefillOutputs,
        SchedulerRunningOutputs,
        SchedulingBudget,
        ScheduledSequenceGroup
    )
class SchedulingStrategy(ABC):
    @abstractmethod
    def schedule_chunked_prefill(self, scheduler: "Scheduler") -> "SchedulerOutputs":
        pass

    @abstractmethod
    def schedule_default(self, scheduler: "Scheduler") -> "SchedulerOutputs":
        pass

    @abstractmethod
    def schedule_swapped(
        self,
        scheduler: "Scheduler",
        budget: "SchedulingBudget",
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> "SchedulerSwappedInOutputs":
        pass
        
    @abstractmethod
    def schedule_prefills(
        self,
        scheduler: "Scheduler",
        budget: "SchedulingBudget",
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> "SchedulerPrefillOutputs":
        pass

    @abstractmethod
    def schedule_running(
        self,
        scheduler: "Scheduler",
        budget: "SchedulingBudget",
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> "SchedulerRunningOutputs":
        pass

class DefaultStrategy(SchedulingStrategy):
    def schedule_chunked_prefill(self, scheduler: "Scheduler") -> "SchedulerOutputs":
        # 默认的chunked_prefill实现
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
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        # Schedule new prefills.
        prefills = self._schedule_prefills(budget,
                                           curr_loras,
                                           enable_chunking=True)

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
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )
    
    def schedule_default(self, scheduler: "Scheduler") -> "SchedulerOutputs":
        # 默认的prefill_default实现
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
        if not self.swapped:
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
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def schedule_swapped(self, scheduler, budget, curr_loras, enable_chunking):
        # 默认的schedule_swapped实现
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
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List["ScheduledSequenceGroup"] = []
        prefill_seq_groups: List["ScheduledSequenceGroup"] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
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

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def schedule_prefills(self, scheduler, budget, curr_loras, enable_chunking):
        # 默认的schedule_prefills实现
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
        seq_groups: List["ScheduledSequenceGroup"] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
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
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

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

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))

    def schedule_running(self, scheduler, budget, curr_loras, enable_chunking):
        # 默认的schedule_running实现
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

        decode_seq_groups: List["ScheduledSequenceGroup"] = ret.decode_seq_groups
        prefill_seq_groups: List["ScheduledSequenceGroup"] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

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

                scheduled_seq_group: "ScheduledSequenceGroup" = \
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
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

class AudioPlayerStrategy(SchedulingStrategy):
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

    def _force_swap_in_by_preemption(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
        blocks_to_swap_out: List[Tuple[int, int]],
        blocks_to_copy: List[Tuple[int, int]],
        budget: "SchedulingBudget",
        is_prefill: bool,
        enable_chunking: bool,
    ) -> Tuple[bool, Optional["ScheduledSequenceGroup"], List[SequenceGroup]]:
        """通过抢占running队列中的序列来强制恢复一个序列
        
        Args:
            seq_group: 需要恢复的序列组
            blocks_to_swap_in: 需要换入的块列表
            blocks_to_swap_out: 需要换出的块列表
            budget: 调度预算
            is_prefill: 是否是prefill阶段
            enable_chunking: 是否启用分块
            
        Returns:
            Tuple[bool, Optional["ScheduledSequenceGroup"], List[SequenceGroup]]: 
                - 是否成功恢复
                - 如果成功,返回调度后的序列组
                - 被抢占的序列组列表
        """
        # print(self.running)
        # 按剩余语音时间降序排序running队列
        self.running = deque(sorted(
            self.running,
            key=lambda x: (
                x.seqs[0].seq_duration - 
                (time.time() - x.metrics.first_scheduled_time)
                if x.metrics else 0
            ),
            reverse=True  # 降序,剩余时间多的优先被抢占
        ))
        #没有确定成功抢占前，running_seqs应该和self.running不指向同一个地址
        #running_seqs = self.running

        # print(self.running)
        # print(running_seqs)
        
        preempted_seqs = []  # 记录被抢占的序列
        running_seqs = deque(self.running)
        # 尝试抢占直到能恢复当前序列
        for victim in running_seqs:
            budget.subtract_num_batched_tokens(seq_group.request_id, 1) # decode一轮产生一个token
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)
            
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

                if (num_new_tokens == 0 or 
                    not budget.can_schedule(
                        num_new_tokens=num_new_tokens,
                        num_new_seqs=num_new_seqs
                    )):
                    return False, None, preempted_seqs
                    
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                scheduled_group = ScheduledSequenceGroup(
                    seq_group, 
                    token_chunk_size=1
                )
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                # self.running = running_seqs #确认成功以后再给self.running重新赋值

                return True, scheduled_group, preempted_seqs
                
        return False, None, preempted_seqs

    def _force_preempt_for_waiting_seq(
        self, 
        waiting_seq: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]]
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
                x.seqs[0].seq_duration - (time.time() - x.metrics.first_scheduled_time) 
                if x.metrics else 0
            ),
            reverse=True
        ))
        # print(f"running: {running_seqs}")
        running_seqs = deque(self.running)
        
        preempted_seqs = []
        
        for victim in running_seqs:
            logger.info(f"目前剩余时间最多的序列有seq_duration:{victim.seqs[0].seq_duration}\n"
                            f"它的completion_token_ids长度为:{len(victim.seqs[0].completion_token_ids)}\n"
                            # f"它的output_text为:{victim.seqs[0].output_text}"
                        )
            # 执行抢占
            self._preempt(victim, blocks_to_swap_out)
            preempted_seqs.append(victim)
            self.running.popleft()
            
            # 检查当前资源是否足够
            if self.block_manager.can_allocate(waiting_seq) == AllocStatus.OK:
                return True, preempted_seqs  # 返回成功状态和被抢占的序列
                
        # 如果抢占所有序列后仍无法分配,则恢复抢占的序列
        # for seq in preempted_seqs:
        #     self.running.append(seq)
            # 注意:_preempt()已经处理了blocks的swap,
            # 所以这里不需要额外处理blocks的恢复
            
        return False, []
    
    def schedule_chunked_prefill(self, scheduler: "Scheduler") -> "SchedulerOutputs":
        # 语音播放器的chunked_prefill实现
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
        )
        curr_loras: Set[int] = set()

        prefills = SchedulerPrefillOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        running_scheduled = self._schedule_running(budget,
                                                   curr_loras,
                                                   enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget, curr_loras)

        # Schedule new prefills.
        prefills = self._schedule_prefills(budget,
                                           curr_loras,
                                           enable_chunking=True)

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
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend([s.seq_group for s in prefills.seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )
    
    def schedule_default(self, scheduler: "Scheduler") -> "SchedulerOutputs":
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
            # scheduled_seq_groups.append("ScheduledSequenceGroup"(seq_group=self.running[len(self.running)-1], token_chunk_size=1))
            # scheduled_seq_groups = []
            # for running_group in self.running:
            #     try:
            #         scheduled_seq_groups.append("ScheduledSequenceGroup"(seq_group=running_group, token_chunk_size=1))
            #     except Exception as e:
            #         print(f"Error creating "ScheduledSequenceGroup" for {running_group}: {e}")

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        all_blocks_to_swap_out = (running_scheduled.blocks_to_swap_out + 
                             prefills.blocks_to_swap_out + swapped_in.blocks_to_swap_out)

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

    def schedule_swapped(self, scheduler, budget, curr_loras, enable_chunking):
        # 基于剩余语音时长的swap调度逻辑
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
        decode_seq_groups: List["ScheduledSequenceGroup"] = []
        prefill_seq_groups: List["ScheduledSequenceGroup"] = []
        infeasible_seq_groups: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []
 
        # print(f"Swapped:{self.swapped}")
        # print(f"running:{self.running}")
        self.swapped = deque(sorted(
            self.swapped,
            key=lambda x: (
                x.seqs[0].seq_duration - (time.time() - x.metrics.first_scheduled_time)
                if x.metrics else float('inf')  # 如果没有metrics,放到最后
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
                remaining_audio_time = (
                    seq_group.seqs[0].seq_duration - 
                    (time.time() - seq_group.metrics.first_scheduled_time)
                    if seq_group.metrics else float('inf')
                )
                print(f"remaining_audio_time: {remaining_audio_time}")
                print(f"seq_group.seqs[0].seq_duration:{seq_group.seqs[0].seq_duration}")
                print(f"seq_group.seqs[0].output_text: {seq_group.seqs[0].output_text}")
                # 如果剩余时间小于阈值,尝试强制恢复
                if remaining_audio_time < 1.0:  # 可配置的阈值
                    print(seq_group.seqs[0].seq_duration)
                    print( (time.time() - seq_group.metrics.first_scheduled_time))
                    print("it's me! help!")
                    success, scheduled_group, preempted_seqs = self._force_swap_in_by_preemption(
                        seq_group,
                        blocks_to_swap_in,
                        blocks_to_swap_out,
                        blocks_to_copy,
                        budget,
                        is_prefill,
                        enable_chunking
                    )
                    # print("Q:are you success?")
                    # print(success)
                    
                    if success:
                        swapped_queue.popleft()
                        decode_seq_groups.append(scheduled_group)
                        swapped_out.extend(preempted_seqs)  # 将被抢占的序列添加到swapped队列
                break  # 无论是否成功抢占,都退出循环

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

    def schedule_prefills(self, scheduler, budget, curr_loras, enable_chunking):
        # 语音播放器的prefills调度逻辑
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
        seq_groups: List["ScheduledSequenceGroup"] = []
        blocks_to_swap_out: List[Tuple[int, int]] = []
        swapped_out: List[SequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            
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
                logger.info("LATER begin")
                # 只对LATER状态检查是否需要强制调度
                if self._should_force_schedule(seq_group, can_allocate):
                    logger.info(
                        f"Attempting force schedule for sequence {seq_group.request_id} "
                        f"(waited for {time.time() - seq_group.arrival_time:.2f}s)"
                    )

                    # fix: 加入对num_new_seqs的计算
                    num_new_seqs = seq_group.get_max_num_running_seqs()
                    if (num_new_tokens == 0
                            or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                    num_new_seqs=num_new_seqs)):
                        break

                    success, preempted = self._force_preempt_for_waiting_seq(
                    seq_group, blocks_to_swap_out)
                    if success:
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
                        # fix: 增加预算
                        budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                        budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                        continue
                    else:
                        logger.warning(
                            f"Failed to force schedule sequence {seq_group.request_id} "
                            f"after waiting for {time.time() - seq_group.arrival_time:.2f}s"
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

    def schedule_running(self, scheduler, budget, curr_loras, enable_chunking):
        # 基于剩余语音时长的running队列调度逻辑
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

        decode_seq_groups: List["ScheduledSequenceGroup"] = ret.decode_seq_groups
        prefill_seq_groups: List[
            "ScheduledSequenceGroup"] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        # 24/11/30 feat: 在running队列里也遵循，按剩余可播放时间排序,优先抢占剩余时间长的序列
        self.running = deque(sorted(
            self.running,
            key=lambda x: (
                x.seqs[0].seq_duration - (time.time() - x.metrics.first_scheduled_time) 
                if x.metrics else 0
            ),
            # reverse=True
        ))

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

                scheduled_seq_group: "ScheduledSequenceGroup" = \
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
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret


# class RateControllerStrategy(SchedulingStrategy):
#     def __init__(self, scheduler: "Scheduler"):
#         self.ttft_slo = scheduler.scheduler_config.ttft_slo
#         self.tbt_slo = scheduler.scheduler_config.tbt_slo
#         self.swap_overhead = 0.08  # 80ms的swap开销
#         self.decode_overhead = 0.2  # 200ms的decode开销
#         self.prefill_rate = 0.6    # 预估prefill速率
#         self.decode_rate = 0.5     # token生成速率
        
#     def _get_waiting_headroom(self, seq_group: SequenceGroup) -> float:
#         """计算waiting请求的余量"""
#         current_time = time.time()
#         wait_time = current_time - seq_group.arrival_time
#         expected_prefill_time = self.prefill_rate  # 可以根据prompt长度调整
        
#         headroom = self.ttft_slo - (wait_time + expected_prefill_time)
#         return headroom
        
#     def _get_running_headroom(self, seq_group: SequenceGroup) -> float:
#         """计算running请求的余量"""
#         current_time = time.time()
#         elapsed_time = current_time - seq_group.metrics.first_scheduled_time
#         num_tokens = len(seq_group.seqs[0].output_token_ids)
        
#         expected_time = num_tokens * self.tbt_slo
#         headroom = expected_time - (elapsed_time + self.decode_overhead)
#         return headroom
        
#     def _get_swapped_headroom(self, seq_group: SequenceGroup) -> float:
#         """计算swapped请求的余量"""
#         running_headroom = self._get_running_headroom(seq_group)
#         # 需要额外考虑swap开销
#         return running_headroom - self.swap_overhead
        
#     def schedule_chunked_prefill(self, scheduler: "Scheduler") -> "SchedulerOutputs":
#         budget = SchedulingBudget(
#             token_budget=scheduler.scheduler_config.max_num_batched_tokens,
#             max_num_seqs=scheduler.scheduler_config.max_num_seqs,
#         )
#         curr_loras = set()
        
#         # 1. 检查是否所有running请求都有足够余量,可以执行非chunked prefill
#         MIN_HEADROOM_FOR_FULL_PREFILL = 1.0  # 可配置阈值
#         all_running_have_enough_headroom = True
#         for seq_group in scheduler.running:
#             if self._get_running_headroom(seq_group) < MIN_HEADROOM_FOR_FULL_PREFILL:
#                 all_running_have_enough_headroom = False
#                 break
                
#         if all_running_have_enough_headroom and scheduler.waiting:
#             # 执行完整的prefill
#             return self.schedule_prefills(scheduler, budget, curr_loras, enable_chunking=False)
            
#         # 2. 处理常规调度情况
#         running_scheduled = self.schedule_running(scheduler, budget, curr_loras, enable_chunking=True)
        
#         # 3. 如果有waiting请求,检查是否需要抢占
#         if scheduler.waiting:
#             waiting_seq = scheduler.waiting[0]
#             waiting_headroom = self._get_waiting_headroom(waiting_seq)
            
#             if waiting_headroom < 0:  # 余量不足,需要考虑抢占
#                 # 寻找可以抢占的running请求(余量充足的)
#                 MIN_VICTIM_HEADROOM = 0.5  # 可配置的最小余量阈值
#                 victims = []
#                 for seq_group in scheduler.running:
#                     if self._get_running_headroom(seq_group) > MIN_VICTIM_HEADROOM:
#                         victims.append(seq_group)
                
#                 if victims:  # 找到了合适的抢占对象
#                     # 执行抢占逻辑...
#                     pass
                    
#         # 4. 处理swap请求
#         swapped_in = SchedulerSwappedInOutputs.create_empty()
#         if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
#             for seq_group in scheduler.swapped:
#                 swapped_headroom = self._get_swapped_headroom(seq_group)
#                 if swapped_headroom < 0:  # 需要紧急换回
#                     # 尝试找到可以抢占的running请求
#                     # 类似上面的抢占逻辑...
#                     pass
            
#             # 处理常规swap in...
#             swapped_in = self.schedule_swapped(scheduler, budget, curr_loras)
            
#         # 5. 最后处理剩余的prefill请求
#         prefills = self.schedule_prefills(scheduler, budget, curr_loras, enable_chunking=True)
        
#         # 合并结果并返回...




# 在Scheduler类中的修改
# class Scheduler:
#     def __init__(self, scheduler_config: SchedulerConfig, ...):
#         strategy_map = {
#             "default": DefaultStrategy(),
#             "audio_player": AudioPlayerStrategy(),
#             "rate_controller": RateControllerStrategy(),
#         }
#         self.strategy = strategy_map[scheduler_config.scheduler_strategy]

#     def _schedule_chunked_prefill(self) -> SchedulerOutputs:
#         return self.strategy.schedule_chunked_prefill(self)

#     def _schedule_swapped(self, budget, curr_loras, enable_chunking):
#         return self.strategy.schedule_swapped(
#             self, budget, curr_loras, enable_chunking)

#     def _schedule_prefills(self, budget, curr_loras, enable_chunking):
#         return self.strategy.schedule_prefills(
#             self, budget, curr_loras, enable_chunking)

#     def _schedule_running(self, budget, curr_loras, enable_chunking):
#         return self.strategy.schedule_running(
#             self, budget, curr_loras, enable_chunking)