import time
from collections import deque
from typing import List, Set, Tuple
from unittest.mock import MagicMock

import pytest  # noqa
from torch import Use  # noqa

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import Scheduler, SchedulingBudget
from vllm.lora.request import LoRARequest
from vllm.sequence import SequenceGroup, SequenceStatus

from .utils import (append_new_token, append_new_token_seq_group,
                    create_dummy_prompt, get_sequence_groups,
                    schedule_and_update_computed_tokens)

def test_force_schedule_and_duration():
    """测试强制调度功能和语音时长计算的完整流程"""
    block_size = 4
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        max_model_len=128,
        max_wait_time=0.5  # 设置较短的等待时间以便测试
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.4,
        swap_space=1,
        cache_dtype="auto"
    )
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # 1. 创建并调度两个running序列
    prompt_tokens = [0] * block_size
    
    # 创建长语音序列(10秒)
    _, long_seq = create_dummy_prompt(
        "1", 
        prompt_length=block_size,
        block_size=block_size,
        prompt_tokens=prompt_tokens
    )
    scheduler.add_seq_group(long_seq)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    append_new_token(out, 1)
    long_seq.seqs[0].seq_duration = 10.0
    long_seq.seqs[0].output_text = "今天天气真不错。"
    
    # 创建短语音序列(3秒)
    _, short_seq = create_dummy_prompt(
        "2",
        prompt_length=block_size,
        block_size=block_size,
        prompt_tokens=prompt_tokens
    )
    scheduler.add_seq_group(short_seq)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    append_new_token(out, 1)
    short_seq.seqs[0].seq_duration = 3.0
    short_seq.seqs[0].output_text = "你好啊。"

    # 2. 创建waiting序列
    _, waiting_seq = create_dummy_prompt(
        "3",
        prompt_length=block_size,
        block_size=block_size,
        prompt_tokens=prompt_tokens
    )
    scheduler.add_seq_group(waiting_seq)
    
    # 3. 第一次调度,waiting_seq应该在等待队列
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert waiting_seq not in get_sequence_groups(out)
    assert waiting_seq in scheduler.waiting
    
    # 4. 等待超过最大等待时间
    time.sleep(0.5)
    
    # 5. 再次调度,这时waiting_seq应该被强制调度
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    scheduled_groups = get_sequence_groups(out)
    
    # 验证调度结果
    assert waiting_seq in scheduled_groups, "waiting_seq应该被调度"
    assert long_seq not in scheduled_groups, "long_seq应该被抢占(剩余时间最长)"
    assert short_seq in scheduled_groups, "short_seq应该继续运行(剩余时间较短)"
    assert out.blocks_to_swap_out, "应该有blocks被换出"
    
    # 验证被抢占的序列进入waiting队列
    assert long_seq in scheduler.waiting, "被抢占的long_seq应该进入waiting队列"
    
    # 6. 继续生成并验证语音时长计算
    if short_seq in scheduler.running:
        seq = short_seq.seqs[0]
        # 继续生成新句子
        prev_duration = seq.seq_duration
        prev_text = seq.output_text
        new_sentence = "今天过得怎么样？"
        seq.output_text = prev_text + new_sentence
        append_new_token(out, 1)
        
        # 验证语音时长计算
        # 1) 总时长应该是所有句子的和
        sentences = [s.strip() for s in seq.output_text.split("。") if s.strip()]
        expected_duration = sum(
            seq.calculate_sentence_duration(s + "。") 
            for s in sentences
        )
        assert abs(seq.seq_duration - expected_duration) < 1e-6
        
        # 2) 新时长应该大于原时长
        assert seq.seq_duration > prev_duration
        
    # 7. 验证新的waiting序列(之前的long_seq)可以再次被调度
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert long_seq in get_sequence_groups(out), "被抢占的序列应该可以再次被调度"

def test_preemption_by_remaining_playback_time():
    """Test if sequences are preempted based on remaining playback time."""
    block_size = 4
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=128,  # 限制批处理token数
        max_num_seqs=4,  # 限制序列数
        max_model_len=128,  #单个序列(包括输入提示和生成的文本)的最大长度限制
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.4,  # 减小GPU内存使用率
        swap_space=1,   #CPU内存中用于交换的空间大小(单位:GB)
        cache_dtype="auto"
    )
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)
    
    # 创建三个序列组,模拟不同的语音时长场景
    seq_groups = []
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i), 
                                         prompt_length=block_size,
                                         block_size=block_size)
        scheduler.add_seq_group(seq_group)
        # 调度并更新计算的tokens
        seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
        append_new_token(out, 1)  # 添加一个token使其进入运行状态
        seq_groups.append(seq_group)
        
    # 设置不同的语音时长和开始时间
    current_time = time.time()
    
    # 序列1: 长语音,刚开始生成 (剩余时间最长)
    seq_groups[0].seqs[0].seq_duration = 10.0  # 10秒语音
    seq_groups[0].metrics.first_scheduled_time = current_time
    
    # 序列2: 中等语音,已生成一段时间
    seq_groups[1].seqs[0].seq_duration = 6.0  # 6秒语音
    seq_groups[1].metrics.first_scheduled_time = current_time - 2.0  # 2秒前开始
    
    # 序列3: 短语音,已生成较长时间
    seq_groups[2].seqs[0].seq_duration = 3.0  # 3秒语音
    seq_groups[2].metrics.first_scheduled_time = current_time - 1.0  # 1秒前开始
    
    # 添加新序列触发抢占
    _, new_seq_group = create_dummy_prompt("4", 
                                         prompt_length=block_size * 2,  # 使用更长的提示以确保需要更多资源
                                         block_size=block_size)
    scheduler.add_seq_group(new_seq_group)
    
    # 执行调度
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    
    # 验证抢占顺序
    preempted_groups = [group for group in seq_groups if group not in scheduler.running]
    assert len(preempted_groups) > 0  # 确保有序列被抢占
    
    if len(preempted_groups) >= 1:
        # 验证剩余时间最长的序列(序列1)被优先抢占
        assert preempted_groups[0].request_id == "0"
        
    if len(preempted_groups) >= 2:
        # 验证剩余时间次长的序列(序列2)被第二个抢占
        assert preempted_groups[1].request_id == "1"

def test_sequence_duration_calculation():
    """Test if sequence duration is correctly calculated when generating tokens."""
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size)
    
    # 创建一个序列组
    prompt = "你好"
    _, seq_group = create_dummy_prompt("1", 
                                     prompt_length=len(prompt), 
                                     block_size=block_size)
    
    # 初始时duration应该为0
    assert seq_group.seqs[0].seq_duration == 0.0
    
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    
    # 获取序列
    seq = seq_group.seqs[0]
    
    # 模拟逐字生成第一个句子 "今天天气不错。"
    test_sentence = "今天天气不错。"
    current_text = ""
    for char in test_sentence:
        prev_duration = seq.seq_duration
        current_text += char
        seq.output_text = current_text
        seq.append_token_id(1, {1: MagicMock(logprob=0.0)})
        
        # 只有在句子结束时才会增加duration
        if seq.is_sentence_end(char):
            expected_duration = prev_duration + seq.calculate_sentence_duration(test_sentence)
            assert abs(seq.seq_duration - expected_duration) < 1e-6
        else:
            assert abs(seq.seq_duration - prev_duration) < 1e-6

    # 记录第一个句子完成后的duration
    first_sentence_duration = seq.seq_duration
    assert abs(first_sentence_duration - seq.calculate_sentence_duration(test_sentence)) < 1e-6

    # 模拟逐字生成第二个句子 "你吃饭了吗？"
    test_sentence = "你吃饭了吗？"
    for char in test_sentence:
        prev_duration = seq.seq_duration
        current_text += char
        seq.output_text = current_text
        seq.append_token_id(1, {1: MagicMock(logprob=0.0)})
        
        # 只有在句子结束时才会增加duration
        if seq.is_sentence_end(char):
            expected_duration = first_sentence_duration + seq.calculate_sentence_duration(test_sentence)
            assert abs(seq.seq_duration - expected_duration) < 1e-6
        else:
            assert abs(seq.seq_duration - prev_duration) < 1e-6

            
def test_scheduler_force_schedule_by_wait_time():
    """测试基于等待时间的强制调度功能"""
    block_size = 4
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=16,  # 减小总的token预算
        max_num_seqs=2,  # 减小最大序列数
        max_model_len=16,
        max_wait_time=0.5
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.4,  # 减小GPU内存使用率
        swap_space=1,
        cache_dtype="auto"
    )
    cache_config.num_cpu_blocks = 4  # 减小CPU块数
    cache_config.num_gpu_blocks = 4  # 减小GPU块数
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # 1. 首先添加并调度一个长序列占用大部分资源
    _, long_seq = create_dummy_prompt(
        "1",
        prompt_length=12,  # 占用大量block
        block_size=block_size,
        best_of=2  # 增加beam search数量来占用更多资源
    )
    scheduler.add_seq_group(long_seq)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups == 1
    append_new_token(out, 1)

    # 2. 添加一个新的序列,此时应该因为资源不足而等待
    _, waiting_seq = create_dummy_prompt(
        "2",
        prompt_length=8,  # 增加prompt长度
        block_size=block_size
    )
    scheduler.add_seq_group(waiting_seq)
    
    # 第一次调度,waiting_seq应该无法调度
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler) 
    assert waiting_seq not in get_sequence_groups(out)
    append_new_token(out, 1)

    # 3. 等待超过最大等待时间
    time.sleep(0.5)
    
    # 4. 再次调度,这时waiting_seq应该被强制调度
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    
    scheduled_groups = get_sequence_groups(out)
    assert waiting_seq in scheduled_groups
    assert long_seq not in scheduled_groups
    assert out.blocks_to_swap_out
    
    # 5. 确认long_seq进入了swapped队列
    assert long_seq in scheduler.swapped

def test_scheduler_add_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(
        100,
        64,
        1,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, cache_dtype="auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq group to scheduler.
    num_seq_group = 4
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           block_size,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
        assert scheduler.get_num_unfinished_seq_groups() == i + 1


def test_scheduler_abort_seq_group():
    block_size = 4
    scheduler_config = SchedulerConfig(
        100,
        64,
        1,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add multiple seq groups to scheduler.
    num_seq_group = 4
    request_ids: Set[str] = set()
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        scheduler.add_seq_group(seq_group)
        request_ids.add(str(i))

    # Abort all added seq groups.
    assert scheduler.get_num_unfinished_seq_groups() == num_seq_group
    scheduler.abort_seq_group(request_ids)
    assert scheduler.get_num_unfinished_seq_groups() == 0


def test_scheduler_schedule_simple():
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(
        64,
        num_seq_group,
        max_model_len,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=block_size,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    append_new_token(out, 1)

    # Schedule seq groups generation.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    append_new_token(out, 1)


def test_scheduler_prefill_prioritized():
    """Verify running batched tokens are not applied to prefill requests."""
    block_size = 4
    max_model_len = 30
    max_batched_num_tokens = 30
    scheduler_config = SchedulerConfig(
        max_batched_num_tokens,
        2,
        max_model_len,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    _, seq_group_a = create_dummy_prompt("1", 1, block_size=block_size)
    scheduler.add_seq_group(seq_group_a)

    # Schedule seq groups prompts.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a]

    # Add a new prefill request B.
    _, seq_group_b = create_dummy_prompt("2", 30, block_size=block_size)
    scheduler.add_seq_group(seq_group_b)

    # Verify prefill requests are prioritized. Since max_batched_num_tokens
    # is 1, new prefill request has to be scheduled first.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_b]


def test_scheduler_schedule_preempt_abort():
    block_size = 4
    max_model_len = 16
    scheduler_config = SchedulerConfig(
        64,
        2,
        max_model_len,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 2
    cache_config.num_gpu_blocks = 2
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # Add seq groups to scheduler.
    seq_a, seq_group_a = create_dummy_prompt("1",
                                             block_size,
                                             block_size=block_size)
    seq_b, seq_group_b = create_dummy_prompt("2",
                                             block_size,
                                             block_size=block_size)
    scheduler.add_seq_group(seq_group_a)
    scheduler.add_seq_group(seq_group_b)

    # Schedule seq groups prompts.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a, seq_group_b]
    assert out.num_batched_tokens == block_size * 2  # seq_a and seq_b
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 2
    assert scheduler.get_num_unfinished_seq_groups() == 2

    # Append "generated" tokens, allowing the sequence to mark prompt tokens as
    # processed.
    append_new_token(out, 1)

    # Schedule seq groups generation and preempt seq group b.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_a]
    assert out.num_batched_tokens == 1
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 2
    assert out.preempted == 1

    # Abort seq group a. Re-schedule seq group b prompt with recomputation.
    scheduler.abort_seq_group("1")
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert get_sequence_groups(out) == [seq_group_b]
    assert out.num_batched_tokens == 5  # 4 prompt + 1 generation.
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == 1
    assert scheduler.get_num_unfinished_seq_groups() == 1


def test_scheduler_max_seqs():
    block_size = 4
    num_seq_group = 4
    max_seq_group = 2
    max_model_len = 16
    scheduler_config = SchedulerConfig(
        64,
        max_seq_group,
        max_model_len,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    all_seq_groups: List[SequenceGroup] = []
    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=block_size,
                                           block_size=block_size)
        all_seq_groups.append(seq_group)

    # Append 1 seq group
    scheduler.add_seq_group(all_seq_groups[0])

    # Schedule seq groups prompts.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])
    append_new_token(out, 1)

    # Schedule seq groups generation.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set([all_seq_groups[0]])
    append_new_token(out, 1)

    # Append 2 more seq group
    scheduler.add_seq_group(all_seq_groups[1])
    scheduler.add_seq_group(all_seq_groups[2])

    # Schedule seq groups prompts.
    # Only 1 seq group should be scheduled since max_seq_group is 2
    # and one is prompting.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set([all_seq_groups[1]])


def test_scheduler_delay_factor():
    block_size = 4
    scheduler_config = SchedulerConfig(
        100,
        64,
        16,
        delay_factor=0.5,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    # schedule first prompt
    seq_group_meta, seq_group = create_dummy_prompt("0",
                                                    prompt_length=block_size,
                                                    block_size=block_size)
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '0'
    append_new_token(out, 1)

    # wait for a second before scheduling next prompt
    time.sleep(1)
    seq_group_meta, seq_group = create_dummy_prompt("1",
                                                    prompt_length=block_size,
                                                    block_size=block_size)
    scheduler.add_seq_group(seq_group)

    # second prompt should *not* be scheduled
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups == 0
    assert seq_group_meta[0].request_id == '0'
    append_new_token(out, 1)

    # wait for more than 0.5 second and try again
    time.sleep(0.6)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_prefill_groups > 0
    assert seq_group_meta[0].request_id == '1'
    append_new_token(out, 1)


def test_swapped_out_prioritized():
    block_size = 4
    scheduler = initialize_scheduler(max_num_seqs=6,
                                     block_size=block_size,
                                     num_cpu_blocks=64,
                                     num_gpu_blocks=64)
    # best_of=2 * 3 == 6 sequences.
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           best_of=2,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 3
    append_new_token(out, 1)

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 2
    assert out.num_batched_tokens == 2
    assert out.blocks_to_swap_out != []
    assert out.blocks_to_swap_in == []
    append_new_token(out, 1)

    # Add 1 more task. Swap should be prioritized over prefill.
    _, seq_group = create_dummy_prompt(str(i),
                                       prompt_length=60,
                                       best_of=2,
                                       block_size=block_size)
    scheduler.add_seq_group(seq_group)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    append_new_token(out, 1)
    assert len(out.scheduled_seq_groups) == 3
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 3
    assert out.blocks_to_swap_in != []
    assert out.blocks_to_swap_out == []


def initialize_scheduler(
    *,
    max_num_seqs=1000,
    max_token_budget=1000,
    max_model_len=1000,
    lora_config=None,
    block_size=4,
    num_cpu_blocks=8,
    num_gpu_blocks=8,
):
    block_size = block_size
    scheduler_config = SchedulerConfig(
        max_token_budget,
        max_num_seqs,
        max_model_len,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = num_cpu_blocks
    cache_config.num_gpu_blocks = num_gpu_blocks
    scheduler = Scheduler(scheduler_config, cache_config, lora_config)
    return scheduler


def create_token_budget(token_budget: int = 10000,
                        max_num_seqs: int = 10000) -> SchedulingBudget:
    return SchedulingBudget(
        token_budget=token_budget,
        max_num_seqs=max_num_seqs,
    )


def add_token_budget(budget: SchedulingBudget,
                     num_batched_tokens: int = 0,
                     num_curr_seqs: int = 0):
    mock_seq_group = create_dummy_prompt('10', prompt_length=60)[1]
    budget.add_num_batched_tokens(mock_seq_group.request_id,
                                  num_batched_tokens)
    budget.add_num_seqs(mock_seq_group.request_id, num_curr_seqs)


def test_prefill_schedule_max_prompt_len():
    """
    Test prompt longer than max_prompt_len is aborted.
    """
    block_size = 4
    scheduler = initialize_scheduler(max_model_len=30, block_size=block_size)
    _, seq_group = create_dummy_prompt("0",
                                       prompt_length=60,
                                       block_size=block_size)
    scheduler.add_seq_group(seq_group)
    budget = create_token_budget()
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 1
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 0


def test_prefill_schedule_token_budget():
    """
    Test token budget respected.
    """
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=64,
                                     num_gpu_blocks=64)
    budget = create_token_budget(token_budget=0)
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)

    # 0 token budget == nothing is scheduled.
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 2

    # 60 token budget == 1 request scheduled.
    budget = create_token_budget(token_budget=60)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 1
    assert budget.num_batched_tokens == 60
    assert budget.num_curr_seqs == 1
    assert len(remaining_waiting) == 1

    # Test when current_batched_tokens respected.
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=16,
                                     num_gpu_blocks=16)
    budget = create_token_budget(token_budget=60)
    add_token_budget(budget, 30, 0)
    _, seq_group = create_dummy_prompt(str(i),
                                       prompt_length=60,
                                       block_size=block_size)
    # Cannot schedule a prompt that doesn't fit the budget.
    scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 30
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 1
    budget = create_token_budget(token_budget=90)
    add_token_budget(budget, 30, 0)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.seq_groups) == 1
    assert budget.num_batched_tokens == 90
    assert budget.num_curr_seqs == 1
    assert len(remaining_waiting) == 0


def test_prefill_schedule_max_seqs():
    """
    Test max seq respected.
    """
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=64,
                                     num_gpu_blocks=64)
    budget = create_token_budget(max_num_seqs=2)
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert budget.num_batched_tokens == 120
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 1

    # Verify curr_num_seqs respected.
    scheduler.waiting = deque()
    budget = create_token_budget(max_num_seqs=2)
    add_token_budget(budget, 0, 2)
    _, seq_group = create_dummy_prompt(str(i),
                                       prompt_length=60,
                                       block_size=block_size)
    scheduler.add_seq_group(seq_group)
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 1


def test_prefill_schedule_max_lora():
    """
    Test max lora is respected and prioritized.
    """
    block_size = 4
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config,
                                     block_size=block_size,
                                     num_cpu_blocks=64,
                                     num_gpu_blocks=64)
    budget = create_token_budget(token_budget=120)
    curr_loras: Set[int] = set()
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_path="abc"))
        scheduler.add_seq_group(seq_group)
    # Add two more requests to verify lora is prioritized.
    # 0: Lora, 1: Lora, 2: regular, 3: regular
    # In the first iteration, index 0, 2 is scheduled.
    # If a request is not scheduled because it hits max lora, it is
    # prioritized. Verify that.
    for i in range(2, 4):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
    # Schedule 2 requests (0 and 2)
    output = scheduler._schedule_prefills(budget, curr_loras)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 2
    assert budget.num_batched_tokens == 120
    assert budget.num_curr_seqs == 2
    assert len(remaining_waiting) == 2
    assert len(curr_loras) == 1
    # The second lora request is scheduled next as FCFS policy.
    # Reset curr_loras so that it can be scheduled.
    curr_loras = set()
    budget = create_token_budget(token_budget=60)
    output = scheduler._schedule_prefills(budget, curr_loras)
    remaining_waiting = scheduler.waiting
    assert len(output.seq_groups) == 1
    assert output.seq_groups[0].seq_group.request_id == "1"
    assert len(remaining_waiting) == 1
    assert len(curr_loras) == 1
    assert budget.num_batched_tokens == 60


def test_prefill_schedule_no_block_manager_capacity():
    """
    Test sequence cannot be scheduled due to block manager has no capacity.
    """
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_gpu_blocks=128,
                                     num_cpu_blocks=128)
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.LATER
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 0
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 3

    scheduler = initialize_scheduler()
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size)
        scheduler.add_seq_group(seq_group)
    scheduler.block_manager.can_allocate = MagicMock()
    scheduler.block_manager.can_allocate.return_value = AllocStatus.NEVER
    output = scheduler._schedule_prefills(budget, None)
    remaining_waiting = scheduler.waiting
    assert len(output.ignored_seq_groups) == 3
    assert len(output.seq_groups) == 0
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(remaining_waiting) == 0


def test_decode_schedule_preempted():
    """
    Test decodes cannot be scheduled and preempted.
    """
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=64,
                                     num_gpu_blocks=64)
    curr_loras = None
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._add_seq_group_to_running(seq_group)
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    # 1 cannot be scheduled, and the lowest priority (request 2)
    # should be preempted. 1 will also be preempted.
    budget = create_token_budget()
    output = scheduler._schedule_running(budget, curr_loras)
    remainig_running = scheduler.running
    assert len(remainig_running) == 0
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert output.decode_seq_groups[0].seq_group.request_id == "0"
    assert len(output.preempted) == 2
    # Verify budgets are updated.
    assert budget.num_batched_tokens == 1
    # NOTE: When enable_chunk is False, num_seqs budget is not updated.
    # assert budget.num_curr_seqs == 1
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == []
    # Nothing is copied.
    assert output.blocks_to_copy == []


def test_decode_swap_beam_search():
    """
    Test best_of > 1 swap out blocks
    """
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_gpu_blocks=64,
                                     num_cpu_blocks=64)
    curr_loras = None
    budget = create_token_budget()
    for i in range(3):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           best_of=2,
                                           block_size=block_size)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._add_seq_group_to_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        budget.add_num_seqs(seq_group.request_id,
                            seq_group.get_max_num_running_seqs())
        budget.add_num_batched_tokens(
            seq_group.request_id, seq_group.num_seqs(SequenceStatus.RUNNING))

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots):
        return seq_group.request_id != "2"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)
    scheduler.block_manager.swap_out = MagicMock()
    expected_swap_mapping = [("5", "7")]
    scheduler.block_manager.swap_out.return_value = expected_swap_mapping

    output = scheduler._schedule_running(budget, curr_loras)
    remainig_running = scheduler.running
    assert len(remainig_running) == 0
    assert len(output.decode_seq_groups) == 2
    assert len(output.prefill_seq_groups) == 0
    assert output.decode_seq_groups[0].seq_group.request_id == "0"
    assert output.decode_seq_groups[1].seq_group.request_id == "1"
    assert len(output.preempted) == 0
    assert len(output.swapped_out) == 1
    # Budget should refledct preempted requests.
    assert budget.num_batched_tokens == 2
    # since there are 2 sequences, 2 should be subtracted.
    assert budget.num_curr_seqs == 4
    # Both should be preempted, not swapped.
    assert output.blocks_to_swap_out == expected_swap_mapping
    # Nothing is copied.
    assert output.blocks_to_copy == []


def test_schedule_decode_blocks_to_copy_update():
    """
    Verify blocks_to_copy is updated.
    """
    block_size = 4
    scheduler = initialize_scheduler(block_size=4,
                                     num_cpu_blocks=16,
                                     num_gpu_blocks=16)
    _, seq_group = create_dummy_prompt("1",
                                       prompt_length=60,
                                       best_of=2,
                                       block_size=block_size)
    curr_loras = None
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(60, seq_group, 1)
    scheduler._add_seq_group_to_running(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.append_slots = MagicMock()
    scheduler.block_manager.append_slots.return_value = [(2, 3)]

    budget = create_token_budget()
    output = scheduler._schedule_running(budget, curr_loras)
    remaining_running = scheduler.running
    assert len(remaining_running) == 0
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert len(output.preempted) == 0
    assert len(output.swapped_out) == 0
    # Nothing is preempted.
    assert output.blocks_to_swap_out == []
    # Since append_slot returns the source -> dist mapping, it should
    # applied.
    assert output.blocks_to_copy == [(2, 3)]


def test_schedule_swapped_simple():
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size)
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    _, seq_group = create_dummy_prompt("1",
                                       prompt_length=4,
                                       best_of=2,
                                       block_size=block_size)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(4, seq_group, 1)
    scheduler._swap_out(seq_group, blocks_to_swap_out)
    scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 0
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    # swap in is the reverse of swap out
    blocks_to_swap_in_reverse = []
    for swapin, swapout in output.blocks_to_swap_in:
        blocks_to_swap_in_reverse.append((swapout, swapin))
    assert blocks_to_swap_out == blocks_to_swap_in_reverse


def test_schedule_swapped_max_token_budget():
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=32,
                                     num_gpu_blocks=32)
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60, best_of=2)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget(token_budget=1)
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0

    # Verify num_batched_tokens are respected.
    budget = create_token_budget(token_budget=1)
    add_token_budget(budget, 1, 0)
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 0
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_schedule_swapped_max_seqs():
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=64,
                                     num_gpu_blocks=64)
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    for i in range(4):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=4)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget(max_num_seqs=2)
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 2
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 2
    assert len(output.prefill_seq_groups) == 0

    # Verify num_curr_seqs are respected.
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 2
    assert budget.num_curr_seqs == 2
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_schedule_swapped_max_loras():
    block_size = 4
    lora_config = LoRAConfig(max_lora_rank=8, max_loras=1)
    scheduler = initialize_scheduler(lora_config=lora_config,
                                     block_size=block_size,
                                     num_cpu_blocks=32,
                                     num_gpu_blocks=32)
    curr_loras: Set[int] = set()
    blocks_to_swap_out: List[Tuple[int, int]] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           block_size=block_size,
                                           lora_request=LoRARequest(
                                               lora_name=str(i),
                                               lora_int_id=i + 1,
                                               lora_path="abc"))
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        scheduler._add_seq_group_to_swapped(seq_group)

    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 1
    assert budget.num_batched_tokens == 1
    assert budget.num_curr_seqs == 1
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert len(curr_loras) == 1


def test_schedule_swapped_cannot_swap_in():
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=32,
                                     num_gpu_blocks=32)
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           best_of=2,
                                           block_size=block_size)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        scheduler._add_seq_group_to_swapped(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.can_swap_in = MagicMock()
    scheduler.block_manager.can_swap_in.return_value = AllocStatus.LATER
    # Since we cannot swap in, none of the requests are swapped in.
    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 2
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_infeasible_swap():
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=32,
                                     num_gpu_blocks=32)
    curr_loras = None
    blocks_to_swap_out: List[Tuple[int, int]] = []
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i),
                                           prompt_length=60,
                                           best_of=2,
                                           block_size=block_size)
        scheduler._allocate_and_set_running(seq_group)
        append_new_token_seq_group(60, seq_group, 1)
        scheduler._swap_out(seq_group, blocks_to_swap_out)
        scheduler._add_seq_group_to_swapped(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.can_swap_in = MagicMock()
    scheduler.block_manager.can_swap_in.return_value = AllocStatus.NEVER
    # Since we cannot swap in, none of the requests are swapped in.
    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 0
    assert len(output.infeasible_seq_groups) == 2
    assert budget.num_batched_tokens == 0
    assert budget.num_curr_seqs == 0
    assert len(output.decode_seq_groups) == 0
    assert len(output.prefill_seq_groups) == 0


def test_schedule_swapped_blocks_to_copy():
    block_size = 4
    scheduler = initialize_scheduler(block_size=block_size,
                                     num_cpu_blocks=32,
                                     num_gpu_blocks=32)
    curr_loras = None
    _, seq_group = create_dummy_prompt("1",
                                       prompt_length=60,
                                       best_of=2,
                                       block_size=block_size)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(60, seq_group, 1)
    blocks_to_swap_out: List[Tuple[int, int]] = []
    scheduler._swap_out(seq_group, blocks_to_swap_out)
    scheduler._add_seq_group_to_swapped(seq_group)

    # The last request should be swapped out.
    scheduler.block_manager.append_slots = MagicMock()
    scheduler.block_manager.append_slots.return_value = [(2, 3)]

    budget = create_token_budget()
    output = scheduler._schedule_swapped(budget, curr_loras)
    remaining_swapped = scheduler.swapped
    assert len(remaining_swapped) == 0
    assert len(output.decode_seq_groups) == 1
    assert len(output.prefill_seq_groups) == 0
    assert output.blocks_to_copy == [(2, 3)]


def test_scheduling_budget():
    TOKEN_BUDGET = 4
    MAX_SEQS = 4
    budget = SchedulingBudget(token_budget=TOKEN_BUDGET, max_num_seqs=MAX_SEQS)
    assert budget.can_schedule(num_new_tokens=1, num_new_seqs=1)
    assert budget.can_schedule(num_new_tokens=4, num_new_seqs=4)
    assert not budget.can_schedule(num_new_tokens=1, num_new_seqs=5)
    assert not budget.can_schedule(num_new_tokens=5, num_new_seqs=1)
    assert not budget.can_schedule(num_new_tokens=5, num_new_seqs=5)
    assert budget.remaining_token_budget() == TOKEN_BUDGET

    # Verify add/subtract num batched tokens.
    _, seq_group = create_dummy_prompt("1", 3)
    budget.add_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 2
    assert budget.num_batched_tokens == 2
    assert budget.can_schedule(num_new_tokens=2, num_new_seqs=1)
    assert not budget.can_schedule(num_new_tokens=3, num_new_seqs=1)
    # Verify adding another seq group is no-op.
    budget.add_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 2
    assert budget.num_batched_tokens == 2
    budget.subtract_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 4
    assert budget.num_batched_tokens == 0
    budget.subtract_num_batched_tokens(seq_group.request_id, 2)
    assert budget.remaining_token_budget() == 4
    assert budget.num_batched_tokens == 0

    # Verify add/subtract max seqs.
    _, seq_group = create_dummy_prompt("1", 3)
    budget.add_num_seqs(seq_group.request_id, 2)
    assert budget.can_schedule(num_new_tokens=1, num_new_seqs=2)
    assert not budget.can_schedule(num_new_tokens=1, num_new_seqs=3)
    assert budget.num_curr_seqs == 2
    # Verify adding another seq group is no-op.
    budget.add_num_seqs(seq_group.request_id, 2)
    assert budget.num_curr_seqs == 2
    budget.subtract_num_seqs(seq_group.request_id, 2)
    assert budget.num_curr_seqs == 0
    budget.subtract_num_seqs(seq_group.request_id, 2)
    assert budget.num_curr_seqs == 0
