#!/usr/bin/env python3
"""
Master orchestration script for all paper experiments and visualizations.

This script runs all 7 experiment/analysis scripts in sequence:
  1. Full-scale evaluation (all models × all tasks)
  2. RISys cluster circularity analysis
  3. ATE protocol format sensitivity analysis
  4. SECURE ceiling effect analysis
  5. Backend inference variance analysis
  6. Framework evaluation visualizations
  7. Comprehensive reports and recommendations

Usage:
  python run_all_experiments.py [--skip-inference] [--subset] [--no-plots]

Options:
  --skip-inference: Skip inference collection (reuse existing results)
  --subset: Run on small subset (n=10 per task) for testing
  --no-plots: Skip visualization generation (faster for CI/testing)
"""

import subprocess
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SCRIPTS = [
    "01_full_scale_evaluation.py",
    "02_risys_cluster_analysis.py",
    "03_ate_protocol_analysis.py",
    "04_secure_ceiling_analysis.py",
    "05_backend_variance_analysis.py",
    "06_generate_framework_visualizations.py",
    "07_generate_reports.py"
]


def run_script(script_name: str, options: dict) -> bool:
    """Run a single analysis script"""
    import sys

    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]

    # Add options from command line
    if options.get("subset"):
        cmd.append("--subset")
    if options.get("no_plots"):
        cmd.append("--no-plots")

    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {script_name}")
    logger.info(f"{'='*70}")

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name} failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {script_name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all paper experiments")
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip full-scale inference (reuse cached)")
    parser.add_argument("--subset", action="store_true",
                       help="Run on small subset for testing")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--only", type=str,
                       help="Run only specific script (by number 1-7)")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CYBERSECURITY LLM BENCHMARK META-EVALUATION")
    logger.info("Master Experiment Orchestration")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("")

    options = {
        "subset": args.subset,
        "no_plots": args.no_plots
    }

    scripts_to_run = SCRIPTS.copy()

    # Filter scripts if --only specified
    if args.only:
        try:
            idx = int(args.only) - 1
            if 0 <= idx < len(SCRIPTS):
                scripts_to_run = [SCRIPTS[idx]]
            else:
                logger.error(f"Invalid script number. Choose 1-{len(SCRIPTS)}")
                sys.exit(1)
        except ValueError:
            logger.error("--only requires a number (1-7)")
            sys.exit(1)

    # Skip inference if requested
    if args.skip_inference:
        scripts_to_run = scripts_to_run[1:]  # Skip script 1
        logger.info("Skipping full-scale inference collection")
        logger.info("")

    # Run scripts
    results = {}
    for script in scripts_to_run:
        success = run_script(script, options)
        results[script] = success

        if not success:
            logger.warning(f"Failed script: {script}")
            # Continue to next script instead of stopping

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for script, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {script}")

    logger.info("")
    logger.info(f"Result: {passed}/{total} scripts completed successfully")

    if passed == total:
        logger.info("")
        logger.info("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        logger.info("")
        logger.info("Output files in ./results/:")
        logger.info("  - Full-scale results: full_scale_results.json")
        logger.info("  - RISys analysis: risys_cluster_analysis.json, risys_circular_evidence.md")
        logger.info("  - ATE analysis: ate_protocol_analysis.json, ate_format_sensitivity.md")
        logger.info("  - SECURE analysis: secure_ceiling_analysis.json, secure_ceiling_evidence.md")
        logger.info("  - Backend variance: backend_variance_analysis.json, backend_variance_report.md")
        logger.info("  - Visualizations: framework_heatmap.png, framework_dimensions.png, etc.")
        logger.info("  - Reports: practitioner_guide.md, model_eval_checklist.md, etc.")
        sys.exit(0)
    else:
        logger.error(f"\n✗ {total - passed} script(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
