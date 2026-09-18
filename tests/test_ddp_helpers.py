from src.utils.ddp import get_rank, get_world_size, is_main_process


def test_without_a_process_group_every_process_is_rank_zero_of_one():
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process()
