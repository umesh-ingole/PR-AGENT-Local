"""Print the artifacts from the latest completed or partially saved pipeline run."""
import json
from pathlib import Path


def main() -> None:
    runs_dir = Path(__file__).resolve().parent / "runs"
    saved_runs = sorted(
        (path for path in runs_dir.glob("*") if path.is_dir()
         and any(path.glob("round_*_findings.json"))),
        key=lambda path: path.name,
    ) if runs_dir.exists() else []

    if not saved_runs:
        print("No saved run found. Run run_pipeline.py first.")
        return

    run_dir = saved_runs[-1]
    print(f"Latest run: {run_dir.name}")
    for findings_path in sorted(run_dir.glob("round_*_findings.json")):
        findings = json.loads(findings_path.read_text(encoding="utf-8"))
        print(f"\nRound {findings['round']} score: {findings['score']}/100")
        print(json.dumps(findings["findings"], indent=2, ensure_ascii=False))

    fixed_files = [path for path in run_dir.glob("*.py")]
    if fixed_files:
        print(f"\nFinal fixed file ({fixed_files[0].name}):")
        print(fixed_files[0].read_text(encoding="utf-8"))
    else:
        print("\nFinal fixed file has not been saved yet.")

    diff_path = run_dir / "changes.diff"
    if diff_path.exists():
        print("Unified diff:")
        print(diff_path.read_text(encoding="utf-8"), end="")
    else:
        print("Unified diff has not been saved yet.")


if __name__ == "__main__":
    main()