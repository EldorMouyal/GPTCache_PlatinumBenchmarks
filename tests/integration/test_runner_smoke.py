# tests/integration/test_runner_smoke.py
import json
from pathlib import Path
import pytest
import yaml

import src.runner as runner


class DummyLLM:
    """Fake LLM whose .invoke() always returns a numeric-looking answer."""
    def invoke(self, prompt: str) -> str:
        return "Answer: 42"


@pytest.fixture(autouse=True)
def patch_build_llm(monkeypatch):
    """
    Ensure the runner never talks to real Ollama.
    Patch both the adapter module and runner symbol (in case of 'from ... import build_llm').
    """
    factory = lambda *a, **k: DummyLLM()
    monkeypatch.setattr("src.models.ollama.build_llm", factory, raising=False)
    monkeypatch.setattr("src.runner.build_llm", factory, raising=False)


@pytest.fixture
def fixture_dataset_path():
    """Path to the tiny platinum subset fixture shipped with the repo."""
    return Path(__file__).parents[1] / "fixtures" / "tiny_platinum_subset.jsonl"


@pytest.fixture
def patch_dataset_loader(monkeypatch, fixture_dataset_path):
    """Patch bench_datasets.platinum.load to read rows from the local JSONL fixture."""
    def fake_load(subset, split, start, limit):
        rows = []
        with fixture_dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        s = int(start or 0)
        l = int(limit or len(rows))
        return rows[s : s + l]

    monkeypatch.setattr("src.bench_datasets.platinum.load", fake_load, raising=True)


def _base_cfg(tmp_path, run_id: str):
    return {
        "run": {"id": run_id},
        "model": {"provider": "ollama", "name": "dummy", "params": {}},
        "dataset": {
            "name": "platinum-bench",
            "subset": "gsm8k",
            "split": "test",
            "slice": {"start": 0, "limit": 3},
        },
        "cache": {"mode": "none"},
        "output": {"dir": str(tmp_path), "filename_pattern": "{run_id}.json"},
    }


def test_run_once_with_dummy_model(patch_dataset_loader, tmp_path):
    cfg = _base_cfg(tmp_path, "smoke-001")
    result = runner.run_once(cfg)

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "items" in result
    assert len(result["items"]) == 3
    for item in result["items"]:
        assert "question" in item
        assert "model_output" in item
        assert "correct" in item


def test_main_writes_json(patch_dataset_loader, tmp_path):
    cfg = _base_cfg(tmp_path, "smoke-002")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    runner.main(str(cfg_path))

    out_path = tmp_path / "smoke-002.json"
    assert out_path.exists(), f"Expected result file not found: {out_path}"

    # Be encoding-agnostic: try utf-8 first, then cp1252 (Windows default).
    raw = out_path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        data = json.loads(raw.decode("cp1252"))

    assert "metrics" in data
    assert "items" in data
    assert len(data["items"]) == 3
