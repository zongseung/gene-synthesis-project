"""WebDataset tar 샤드에서 약초 이미지를 논리경로로 랜덤 액세스해 읽는다.

트레이너가 기대하는 논리경로(예: '151/001_칡/x.jpg')를
샤드 인덱스(herb_shard_index.json: 논리경로 → [tar_basename, member])로 해석해
해당 tar에서 바이트를 추출 → PIL Image 반환. 디스크 언팩 불필요.

스레드/멀티프로세스 주의:
- tarfile 핸들은 fork-안전하지 않음 → __init__에서 열지 않고 **워커별 lazy open**.
- getmembers()는 샤드당 1회(캐시). 이후 extractfile은 오프셋 seek라 빠름.
- intra-process는 Lock으로 보호(num_workers=0 또는 thread 사용 시).
"""
from __future__ import annotations
import io, json, os, tarfile, threading


class ShardImageReader:
    def __init__(self, index_path: str, shard_dir: str):
        with open(index_path, encoding="utf-8") as f:
            self.index = json.load(f)          # logical -> [tar_base, member]
        self.shard_dir = shard_dir
        self._tars: dict = {}                   # tar_base -> (tarfile, {member: TarInfo})
        self._lock = threading.Lock()

    def __contains__(self, logical: str) -> bool:
        return logical in self.index

    def _get_tar(self, base: str):
        t = self._tars.get(base)
        if t is None:
            with self._lock:
                t = self._tars.get(base)
                if t is None:
                    tf = tarfile.open(os.path.join(self.shard_dir, base))
                    members = {}
                    try:                       # 612 tar 8개는 잘려 있다 → 읽은 데까지 사용
                        for m in tf:
                            members[m.name] = m
                    except (tarfile.ReadError, EOFError):
                        pass
                    t = (tf, members)
                    self._tars[base] = t
        return t

    def get(self, logical: str):
        """logical 경로 → PIL.Image(RGB) 또는 None(인덱스/멤버 없음)."""
        ent = self.index.get(logical)
        if ent is None:
            return None
        base, member = ent
        tf, members = self._get_tar(base)
        info = members.get(member)
        if info is None:
            return None
        with self._lock:
            fobj = tf.extractfile(info)
            data = fobj.read()
        from PIL import Image
        return Image.open(io.BytesIO(data)).convert("RGB")
