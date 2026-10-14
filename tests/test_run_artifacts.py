import json

from run_pipeline import save_final_artifacts, save_round_findings


def test_run_artifacts_save_round_findings_and_unified_diff(tmp_path):
    update = {
        "score": 72,
        "passed": False,
        "merged": [{"line": 1, "severity": "medium", "title": "Example"}],
    }
    findings_path = save_round_findings(tmp_path, 1, update)
    findings = json.loads(findings_path.read_text(encoding="utf-8"))

    assert findings["round"] == 1
    assert findings["score"] == 72
    assert findings["findings"] == update["merged"]

    original = "value = 1\n"
    fixed = "value = 2\n"
    fixed_path, diff_path = save_final_artifacts(
        tmp_path, "sample.py", original, fixed)

    assert fixed_path.read_text(encoding="utf-8") == fixed
    diff = diff_path.read_text(encoding="utf-8")
    assert "--- original/sample.py" in diff
    assert "+++ fixed/sample.py" in diff
    assert "-value = 1" in diff
    assert "+value = 2" in diff