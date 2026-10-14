"""Run the sample review pipeline without starting the Streamlit UI."""
import json
import difflib
import os
import time
from datetime import datetime
from pathlib import Path

from config import EXT_TO_LANG, MAX_ITERATIONS
from graph import build_graph


def create_run_directory(runs_dir: Path) -> Path:
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_round_findings(run_dir: Path, round_number: int, update: dict) -> Path:
    findings_path = run_dir / f"round_{round_number:02d}_findings.json"
    findings_path.write_text(
        json.dumps({
            "round": round_number,
            "score": update["score"],
            "passed": update.get("passed", False),
            "findings": update.get("merged", []),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return findings_path


def save_final_artifacts(run_dir: Path, filename: str, original: str, fixed: str) -> tuple[Path, Path]:
    fixed_path = run_dir / filename
    fixed_path.write_text(fixed, encoding="utf-8")

    diff_path = run_dir / "changes.diff"
    diff_text = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        fixed.splitlines(keepends=True),
        fromfile=f"original/{filename}",
        tofile=f"fixed/{filename}",
    ))
    diff_path.write_text(diff_text, encoding="utf-8")
    return fixed_path, diff_path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    sample_path = project_dir / "sample" / "buggy_sample.py"
    with sample_path.open(encoding="utf-8") as sample_file:
        code = sample_file.read()

    filename = sample_path.name
    language = EXT_TO_LANG[os.path.splitext(filename)[1].lower()]
    run_dir = create_run_directory(project_dir / "runs")
    print(f"Saving run artifacts to: {run_dir}")
    initial = {
        "filename": filename,
        "language": language,
        "original_code": code,
        "current_code": code,
        "force_local": True,
        "max_iterations": MAX_ITERATIONS,
        "providers": [],
        "log": [],
        "score_history": [],
    }
    final = dict(initial)
    started = time.perf_counter()
    graph = build_graph()
    score_round = 0

    for chunk in graph.stream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            elapsed = time.perf_counter() - started
            for line in update.get("log", []):
                print(f"[{elapsed:7.2f}s] {node}: {line}")
            if "score" in update:
                score_round += 1
                print(f"[{elapsed:7.2f}s] Score after round {score_round}: {update['score']}/100")
                findings_path = save_round_findings(run_dir, score_round, update)
                print(f"[{elapsed:7.2f}s] Saved round findings: {findings_path.name}")
            for key, value in update.items():
                if key in ("providers", "log", "score_history"):
                    final[key] = final.get(key, []) + value
                else:
                    final[key] = value

    fixed_path, diff_path = save_final_artifacts(
        run_dir, filename, initial["original_code"], final["current_code"])
    print(f"\nFinal score: {final['score']}/100")
    print("Final findings:")
    if final["merged"]:
        print(json.dumps(final["merged"], indent=2, ensure_ascii=False))
    else:
        print("No findings.")
    print(f"Final fixed file: {fixed_path}")
    print(f"Unified diff: {diff_path}")
    print(f"\nTotal elapsed: {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()