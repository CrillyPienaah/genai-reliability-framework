#!/usr/bin/env python3
"""
scripts/parse_ci_report.py
───────────────────────────
Reads a ci_report.json and sets GitHub Actions output variable `gate_passed`.
Exits with code 1 if the gate failed — GitHub Actions treats non-zero as failure.

Called by .github/workflows/eval.yml after the eval run.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to ci_report.json")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"::error::Report file not found: {report_path}")
        return 1

    report = json.loads(report_path.read_text())
    passed = report.get("ci_gate_passed", False)

    # Set GitHub Actions output variable
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"gate_passed={'true' if passed else 'false'}\n")
    else:
        # Local run — just print
        print(f"gate_passed={'true' if passed else 'false'}")

    if passed:
        print("✅ Eval CI gate PASSED")
        return 0
    else:
        failures = report.get("gate_failures", [])
        print(f"❌ Eval CI gate FAILED — {len(failures)} regression(s) detected:")
        for failure in failures:
            print(f"   - {failure}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
