"""Run the sample review pipeline without starting the Streamlit UI."""
import argparse
import json
import difflib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from config import EXT_TO_LANG, MAX_ITERATIONS
import agents
from graph import build_graph
import llm_gateway
from static_analysis import static_analysis_mode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use Gemini first with Ollama fallback and allow three fix rounds.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="With --cloud, do not fall back to Ollama if Gemini fails.",
    )
    return parser.parse_args(argv)


def pipeline_settings(cloud: bool, no_fallback: bool = False) -> dict:
    provider_order = "gemini" if cloud and no_fallback else "gemini,ollama" if cloud else "ollama"
    mode = (
        "cloud (Gemini only)" if cloud and no_fallback
        else "cloud (Gemini first, Ollama fallback)" if cloud
        else "local-only (Ollama)"
    )
    return {
        "force_local": not cloud,
        "max_iterations": MAX_ITERATIONS if cloud else 2,
        "provider_order": provider_order,
        "mode": mode,
        "no_fallback": cloud and no_fallback,
    }


def cloud_startup_warning(cloud: bool, no_fallback: bool = False) -> str | None:
    if cloud and not os.getenv("GOOGLE_API_KEY"):
        if no_fallback:
            return "GOOGLE_API_KEY not set: --cloud --no-fallback cannot call Gemini."
        return "GOOGLE_API_KEY not set: --cloud will fall back to the local model and be slow."
    return None


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
            "raw_penalty": update["raw_penalty"],
            "passed": update.get("passed", False),
            "dropped_unverified_count": update.get("dropped_unverified_count", 0),
            "findings": update.get("merged", []),
            "warnings": [
                finding for finding in update.get("merged", [])
                if finding.get("agent") in {"llm-unavailable", "agent-response-error"}
            ],
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return findings_path


def build_run_info(
    settings: dict,
    final_state: dict,
    start_time: str,
    total_elapsed_seconds: float,
    termination_reason: str,
) -> dict:
    return {
        "providers": list(final_state.get("providers", [])),
        "models": {
            "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
        },
        "cloud": not settings["force_local"],
        "force_local": settings["force_local"],
        "max_iterations": settings["max_iterations"],
        "start_time": start_time,
        "total_elapsed_seconds": round(total_elapsed_seconds, 3),
        "termination_reason": termination_reason,
        "static_analysis_mode": settings["static_analysis_mode"],
        "best_round": final_state.get("best_round"),
        "raw_penalty_history": list(final_state.get("penalty_history", [])),
        "rejected_fix_count": final_state.get("rejected_fix_count", 0),
        "llm_call_count": final_state.get("llm_call_count", 0),
        "dropped_unverified_history": list(
            final_state.get("dropped_unverified_history", [])
        ),
    }


def save_run_info(run_dir: Path, run_info: dict) -> Path:
    info_path = run_dir / "run_info.json"
    info_path.write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return info_path


def termination_reason(final_state: dict, max_iterations: int) -> str:
    if final_state.get("passed", False):
        return "passed"
    if final_state.get("stop_no_improvement", False):
        return "early stop"
    if final_state.get("iteration", 0) >= max_iterations:
        return "round cap"
    return "error"


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


def select_best_round(
    complete_rounds: list[dict], last_code: str, last_round: int,
) -> tuple[str, int | None, str]:
    if not complete_rounds:
        if last_round:
            return (
                last_code,
                last_round,
                f"No complete review round; using last round {last_round}.",
            )
        return last_code, None, "No review rounds completed; using original code."
    best = max(
        complete_rounds,
        key=lambda item: (-item["raw_penalty"], item["round"]),
    )
    return (
        best["code"],
        best["round"],
        f"Selected round {best['round']} with lowest complete-round raw penalty "
        f"({best['raw_penalty']}); ties go to the later round.",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = pipeline_settings(args.cloud, args.no_fallback)
    settings["static_analysis_mode"] = static_analysis_mode()
    print(f"Static analysis: {settings['static_analysis_mode']}")
    if not settings["force_local"]:
        os.environ["LLM_PROVIDERS"] = settings["provider_order"]
    print(f"Mode: {settings['mode']}; max_iterations={settings['max_iterations']}")
    warning = cloud_startup_warning(
        not settings["force_local"], settings["no_fallback"]
    )
    if warning:
        print(f"Warning: {warning}")

    project_dir = Path(__file__).resolve().parent
    sample_path = project_dir / "sample" / "buggy_sample.py"
    with sample_path.open(encoding="utf-8") as sample_file:
        code = sample_file.read()

    filename = sample_path.name
    language = EXT_TO_LANG[os.path.splitext(filename)[1].lower()]
    run_dir = create_run_directory(project_dir / "runs")
    print(f"Saving run artifacts to: {run_dir}")
    start_time = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    llm_call_count = 0

    def observe_llm_call(
        stage: str,
        provider: str,
        model: str,
        duration: float,
        success: bool | None,
        error_detail: str = "",
    ) -> None:
        nonlocal llm_call_count
        if stage == "start":
            llm_call_count += 1
            print(
                f"[{time.perf_counter() - started:7.2f}s] "
                f"LLM call {llm_call_count} started: {provider}/{model}"
            )
        else:
            outcome = "success" if success else "failed"
            print(
                f"[{time.perf_counter() - started:7.2f}s] "
                f"LLM call finished: {provider}/{model}; {outcome}; "
                f"call elapsed {duration:.2f}s"
                + (f"; {error_detail}" if error_detail else "")
            )

    initial = {
        "filename": filename,
        "language": language,
        "original_code": code,
        "current_code": code,
        "run_dir": str(run_dir),
        "force_local": settings["force_local"],
        "max_iterations": settings["max_iterations"],
        "providers": [],
        "log": [],
        "score_history": [],
        "penalty_history": [],
        "rejected_fix_count": 0,
        "response_errors": [],
        "dropped_finding_counts": [],
        "dropped_unverified_history": [],
        "complete_rounds": [],
        "dropped_finding_count_index": 0,
        "review_incomplete": False,
    }
    final = dict(initial)
    score_round = 0
    previous_observer = llm_gateway.set_call_observer(observe_llm_call)
    try:
        graph = build_graph()
        for chunk in graph.stream(initial, stream_mode="updates"):
            for node, update in chunk.items():
                if not update:
                    continue
                elapsed = time.perf_counter() - started
                for line in update.get("log", []):
                    print(f"[{elapsed:7.2f}s] {node}: {line}")
                if "score" in update:
                    score_round += 1
                    print(
                        f"[{elapsed:7.2f}s] Score after round {score_round}: "
                        f"{update['score']}/100; raw penalty: {update['raw_penalty']}"
                    )
                    findings_path = save_round_findings(run_dir, score_round, update)
                    print(f"[{elapsed:7.2f}s] Saved round findings: {findings_path.name}")
                for key, value in update.items():
                    if key in (
                        "providers", "log", "score_history", "penalty_history",
                        "rejected_fix_count", "response_errors", "dropped_finding_counts",
                        "dropped_unverified_history", "complete_rounds",
                    ):
                        default = 0 if key == "rejected_fix_count" else []
                        final[key] = final.get(key, default) + value
                    else:
                        final[key] = value

        best_code, best_round, best_reason = select_best_round(
            final.get("complete_rounds", []),
            final["current_code"],
            score_round,
        )
        final["current_code"] = best_code
        final["best_round"] = best_round
        print(f"Best round: {best_reason}")
        fixed_path, diff_path = save_final_artifacts(
            run_dir, filename, initial["original_code"], best_code)
    except Exception:
        elapsed = time.perf_counter() - started
        save_run_info(
            run_dir, build_run_info(
                settings,
                {**final, "llm_call_count": llm_call_count},
                start_time,
                elapsed,
                "error",
            ),
        )
        raise
    finally:
        llm_gateway.set_call_observer(previous_observer)

    elapsed = time.perf_counter() - started
    reason = termination_reason(final, settings["max_iterations"])
    save_run_info(
        run_dir,
        build_run_info(
            settings, {**final, "llm_call_count": llm_call_count},
            start_time, elapsed, reason,
        ),
    )
    banner = agents.review_incomplete_banner(final)
    if banner:
        print(banner)
    print(
        f"\nFinal score: {final['score']}/100; "
        f"raw penalty: {final['raw_penalty']}"
    )
    print("Final findings:")
    warnings = [
        finding for finding in final["merged"]
        if finding.get("agent") in {"llm-unavailable", "agent-response-error"}
    ]
    findings = [finding for finding in final["merged"] if finding not in warnings]
    if findings:
        print(json.dumps(findings, indent=2, ensure_ascii=False))
    else:
        print("No findings.")
    if warnings:
        print("Warnings:")
        print(json.dumps(warnings, indent=2, ensure_ascii=False))
    print(f"Final fixed file: {fixed_path}")
    print(f"Unified diff: {diff_path}")
    print(f"\nTotal elapsed: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()