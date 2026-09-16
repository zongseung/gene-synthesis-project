import ast
import json
from pathlib import Path

from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
NOTEBOOK = WORKSPACE / "notebooks" / "vlm_initial_evaluation.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "hanmed"
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook_runs_model_in_kernel_and_reuses_quick_eval():
    source = "\n".join(_code_cells())
    assert "scripts/quick_eval_sft_varco.py" in source
    assert "quick_eval._load_model(" in source
    assert "def ask(" in source
    assert "display(" in source
    assert "subprocess" not in source
    assert "pip install" not in source
    for code in _code_cells():
        ast.parse(code)


def test_ask_shows_image_question_and_answer(tmp_path, capsys):
    cells = _code_cells()
    helper = next(code for code in cells if "def ask(" in code)

    class FakeQuickEval:
        calls = []

        @staticmethod
        def _generate(processor, model, question, image=None):
            FakeQuickEval.calls.append((question, image))
            return "가짜 답변", 0.5

        @staticmethod
        def _load_images(rows, cfg):
            return {row["id"]: Image.new("RGB", (2, 2)) for row in rows}

    namespace = {
        "quick_eval": FakeQuickEval,
        "cfg": {},
        "processor": None,
        "model": None,
    }
    exec(compile(helper, str(NOTEBOOK), "exec"), namespace)
    ask = namespace["ask"]

    assert ask("문헌만 묻는 질문") == "가짜 답변"
    assert FakeQuickEval.calls[-1] == ("문헌만 묻는 질문", None)

    local = tmp_path / "leaf.jpg"
    Image.new("RGB", (4, 4)).save(local)
    assert ask("이 식물은?", image=str(local)) == "가짜 답변"
    assert FakeQuickEval.calls[-1][1].size == (4, 4)

    assert ask("설진 소견은?", image="shezhenv3/val/A (1).jpg") == "가짜 답변"
    assert FakeQuickEval.calls[-1][1].size == (2, 2)

    out = capsys.readouterr().out
    assert "문헌만 묻는 질문" in out
    assert "가짜 답변" in out
