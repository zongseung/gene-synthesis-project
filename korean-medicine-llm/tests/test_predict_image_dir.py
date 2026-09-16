import importlib.util
import sys
from pathlib import Path


sys.path.insert(0, str(Path("scripts").resolve()))
_spec = importlib.util.spec_from_file_location(
    "predict_image_dir", Path("scripts/predict_image_dir.py")
)
predict_image_dir = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_image_dir)

compact_records = predict_image_dir.compact_records
discover_images = predict_image_dir.discover_images
pending_images = predict_image_dir.pending_images
prompt_for = predict_image_dir.prompt_for


def test_directory_prediction_helpers(tmp_path: Path):
    first = tmp_path / "a.JPG"
    second = tmp_path / "nested" / "b.png"
    second.parent.mkdir()
    first.touch()
    second.touch()
    (tmp_path / "ignore.txt").touch()

    images = discover_images(tmp_path)
    assert images == [first.resolve(), second.resolve()]
    assert "이름" in prompt_for("herb")
    assert "관찰" in prompt_for("tongue")
    assert "변증하지" in prompt_for("tongue")

    records = [
        {"image": str(first.resolve()), "task": "herb", "answer_text": "칡"},
        {"image": str(second.resolve()), "task": "herb", "error": "broken"},
    ]
    assert pending_images(images, records, "herb") == [second.resolve()]
    assert pending_images(images, records, "tongue") == images
    retried = {**records[1], "answer_text": "황기", "error": None}
    assert compact_records(records + [retried])[-1]["answer_text"] == "황기"
