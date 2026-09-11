"""Sequential GPU smoke -> complete tests -> v27 reference audit -> 52 pilot jobs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/femba_reference_free_token_pilot_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    launch = OUTPUT / "launches" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONHASHSEED="2001", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", TMPDIR=str(OUTPUT / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_at": now(), "supervisor_pid": os.getpid(),
              "pilot_jobs": 52, "smoke_jobs": 52, "outer_test_scored": False, "EA_allowed": True, "stages": []}
    def save():
        temp = launch / "status.tmp.json"
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(launch / "status.json")
    command = "scripts/run_femba_reference_free_token_pilot.py"
    stages = [("preflight", [command, "--stage", "preflight"]),
              ("smoke", [command, "--smoke", "--resume"]),
              ("tests", ["-m", "pytest", "-q", "--capture=sys", "--basetemp=" + str(OUTPUT / "acceptance_test_tmp")]),
              ("v27_reference", ["scripts/verify_reproduction.py"]),
              ("pilot", [command, "--stage", "all", "--resume"])]
    save()
    try:
        for name, args in stages:
            entry = {"stage": name, "started_at": now()}
            status["stages"].append(entry)
            status["current_stage"] = name
            with (launch / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
                status["pid"] = child.pid
                save()
                code = child.wait()
            entry.update(exit_code=code, finished_at=now())
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}; see {name}.log")
            if name in ("smoke", "pilot"):
                mode = "smoke" if name == "smoke" else "pilot"
                report = json.loads((OUTPUT / mode / "seed_2001/summary/aggregate_report.json").read_text())
                if not report["full_matrix_completed"] or report["completed_jobs"] != 52 or report["outer_test_scored"]:
                    raise RuntimeError("Incomplete pilot/smoke matrix")
                if name == "smoke" and not (report["all_two_step_checks"] and report["all_online_checks"]):
                    raise RuntimeError("GPU smoke acceptance failed")
            save()
        status.update(status="complete", finished_at=now(), exit_code=0)
        save()
    except BaseException:
        status.update(status="failed", finished_at=now(), error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
