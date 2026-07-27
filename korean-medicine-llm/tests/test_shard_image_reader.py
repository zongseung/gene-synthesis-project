"""S0a — 잘린 612 tar 에서도 리더가 예외 없이 유효 멤버를 반환한다 (D1)."""
import pytest

from hanmed_mm.data.shard_image_reader import ShardImageReader

INDEX = "data/shards/herb_shard_index.json"
SHARD_DIR = "data/shards"
# 612_train_w00_000000.tar 는 잘려 있다 (tarfile.getmembers → ReadError).
TRUNCATED_LOGICAL = "612/가는장구채/가는장구채_꽃_1313661.jpg"


@pytest.mark.skipif(not __import__("os").path.exists(INDEX), reason="샤드 인덱스 없음")
def test_reads_from_truncated_tar():
    r = ShardImageReader(INDEX, SHARD_DIR)
    img = r.get(TRUNCATED_LOGICAL)
    assert img is not None
    assert img.mode == "RGB"
