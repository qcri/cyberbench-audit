#!/usr/bin/env python3
"""
Generate comprehensive report and practitioner recommendation matrices.

Creates actionable outputs for different audiences:
1. Practitioners: Which benchmarks to use for their task?
2. Benchmark Authors: How to improve evaluation rigor?
3. Model Developers: What are the quality standards?
4. Reviewers: Standardized assessment checklist

Outputs:
- results/practitioner_guide.md
- results/benchmark_recommendations.csv
- results/model_eval_checklist.md
- results/author_improvement_roadmap.md
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def generate_practitioner_guide() -> None:
    """Generate guide for practitioners choosing benchmarks"""

    guide = []
    guide.append("# Practitioner's Guide to Cybersecurity LLM Benchmarks")
    guide.append("")

    guide.append("## Quick Reference: Which Benchmark for My Use Case?")
    guide.append("")

    recommendations = [
        {
            "use_case": "Model Selection (General)",
            "recommended": ["SecBench", "CyberMetric", "CISSP"],
            "rationale": "High validation provenance (D2), diverse tasks (D3)",
            "avoid": "CTI-Bench alone (RISys cluster)",
            "priority": "⭐⭐⭐"
        },
        {
            "use_case": "CTI/Threat Intelligence Tasks",
            "recommended": ["AthenaBench (ATE fixed)", "SecBench"],
            "rationale": "Structured extraction tasks, but use LLM judge (not regex)",
            "avoid": "Original CTI-Bench ATE (regex evaluation broken)",
            "priority": "⭐⭐⭐"
        },
        {
            "use_case": "ICS/OT Security",
            "recommended": ["SECURE (exclude MAET/CWET/KCV)"],
            "rationale": "Domain-specific, but MAET/CWET/KCV have ceiling effects",
            "avoid": "Using SECURE ceiling tasks as primary metrics",
            "priority": "⭐⭐"
        },
        {
            "use_case": "Knowledge Assessment (MCQ)",
            "recommended": ["CyberMetric", "CISSP", "MMLU-CS"],
            "rationale": "Validated, reproducible, no ceiling effects",
            "avoid": "RedSage-Bench alone (benchmark-specific tuning concern)",
            "priority": "⭐⭐⭐"
        },
        {
            "use_case": "Multilingual Evaluation",
            "recommended": ["SecBench (Chinese)", "SEvenLLM-Bench"],
            "rationale": "Only benchmarks with non-English coverage",
            "avoid": "Most benchmarks (English-only)",
            "priority": "⭐⭐⭐ (if needed)"
        },
        {
            "use_case": "Model Comparison (Fairness Check)",
            "recommended": ["Your own unified pipeline (reference harness provided)"],
            "rationale": "No existing papers compare PRIMUS vs RedSage vs Foundation-Sec fairly",
            "avoid": "Comparing results across papers (different harnesses, backends)",
            "priority": "⭐⭐⭐"
        }
    ]

    for rec in recommendations:
        guide.append(f"### {rec['priority']} {rec['use_case']}")
        guide.append("")
        guide.append(f"**Recommended:** {', '.join(rec['recommended'])}")
        guide.append(f"**Rationale:** {rec['rationale']}")
        guide.append(f"**Avoid:** {rec['avoid']}")
        guide.append("")

    guide.append("## Critical Issues You Should Know About")
    guide.append("")

    issues = [
        {
            "benchmark": "CTI-Bench ATE",
            "issue": "ATE evaluation uses strict regex on last line only",
            "impact": "Reports 0% F1 due to format sensitivity, not model capability",
            "action": "Use LLM-as-Judge or fixed evaluation pipeline"
        },
        {
            "benchmark": "SECURE MAET/CWET/KCV",
            "issue": "Base Llama-3.1-8B achieves 100% accuracy",
            "impact": "Cannot discriminate between base and security-specialized models",
            "action": "Exclude from model comparisons or use as sanity check only"
        },
        {
            "benchmark": "CTI-Bench + AthenaBench + SECURE",
            "issue": "All from RISys-Lab. RedSage is also from RISys-Lab",
            "impact": "Correlated evaluation signals; models tuned for this cluster inflated",
            "action": "Treat as single cluster; weight independently benchmarks equally"
        },
        {
            "benchmark": "All benchmarks",
            "issue": "Evaluation backend (HF vs vLLM) changes results",
            "impact": "57.89% (HF) vs 47.50% (vLLM) on same responses",
            "action": "Always use HuggingFace Transformers; document backend version"
        },
        {
            "benchmark": "All except SecBench, SEvenLLM",
            "issue": "No Arabic language support",
            "impact": "Cannot evaluate models on primary threat actor language",
            "action": "Add Arabic evaluation if applicable to your domain"
        }
    ]

    guide.append("| Benchmark | Issue | Impact | Action |")
    guide.append("|-----------|-------|--------|--------|")
    for issue in issues:
        guide.append(
            f"| {issue['benchmark']} | {issue['issue']} | {issue['impact']} | {issue['action']} |"
        )
    guide.append("")

    guide.append("## Evaluation Protocol Checklist")
    guide.append("")
    guide.append("Before publishing results using any cybersecurity LLM benchmark:")
    guide.append("")
    guide.append("- [ ] **Harness**: Document exact evaluation harness (lm-eval? lighteval? custom?)")
    guide.append("- [ ] **Backend**: State inference backend (HF Transformers? vLLM?) and versions")
    guide.append("- [ ] **Judge**: If using LLM judge, report model/temperature/version")
    guide.append("- [ ] **Genealogy**: Disclose if benchmark authors overlap with model developers")
    guide.append("- [ ] **Clustering**: Group RISys-Lab benchmarks as single signal")
    guide.append("- [ ] **ATE Fix**: If using ATE, explicitly use LLM judge (not regex)")
    guide.append("- [ ] **SECURE**: Exclude MAET/CWET/KCV from aggregate scores or explain ceiling")
    guide.append("- [ ] **Confidence Intervals**: Report with $n \\geq 100$ per task")
    guide.append("- [ ] **Multilingual**: Include at least one non-English evaluation if claiming multilingual")
    guide.append("")

    with open(OUTPUT_DIR / "practitioner_guide.md", 'w') as f:
        f.write('\n'.join(guide))

    logger.info("✓ Practitioner guide saved")


def generate_benchmark_recommendations_csv() -> None:
    """Generate CSV ranking benchmarks by quality criteria"""

    benchmarks = [
        {
            "benchmark": "CyberMetric",
            "scale": "M",
            "validation": "H",
            "diversity": "L",
            "eval_robust": "M",
            "repro": "H",
            "multi": "N",
            "overall_score": "18/30",
            "best_for": "Knowledge assessment with high-quality annotations",
            "concerns": "Limited task diversity"
        },
        {
            "benchmark": "SecBench",
            "scale": "H",
            "validation": "H",
            "diversity": "M",
            "eval_robust": "M",
            "repro": "M",
            "multi": "M",
            "overall_score": "24/30",
            "best_for": "Comprehensive evaluation, multilingual options",
            "concerns": "Some synthetic generation"
        },
        {
            "benchmark": "CISSP",
            "scale": "M",
            "validation": "L",
            "diversity": "L",
            "eval_robust": "H",
            "repro": "N",
            "multi": "N",
            "overall_score": "13/30",
            "best_for": "Professional certification benchmark",
            "concerns": "Not public, copyright issues"
        },
        {
            "benchmark": "CTI-Bench",
            "scale": "M",
            "validation": "N",
            "diversity": "H",
            "eval_robust": "L",
            "repro": "M",
            "multi": "N",
            "overall_score": "12/30",
            "best_for": "N/A (broken evaluation protocol)",
            "concerns": "RISys-Lab cluster, ATE regex broken, validation undisclosed"
        },
        {
            "benchmark": "AthenaBench",
            "scale": "H",
            "validation": "N",
            "diversity": "H",
            "eval_robust": "L",
            "repro": "M",
            "multi": "N",
            "overall_score": "15/30",
            "best_for": "N/A (inherits CTI-Bench issues)",
            "concerns": "RISys-Lab cluster, same evaluation bugs"
        },
        {
            "benchmark": "SECURE",
            "scale": "N",
            "validation": "N",
            "diversity": "M",
            "eval_robust": "M",
            "repro": "L",
            "multi": "N",
            "overall_score": "9/30",
            "best_for": "N/A (ceiling effects)",
            "concerns": "RISys-Lab cluster, MAET/CWET/KCV ceiling effects"
        },
        {
            "benchmark": "RedSage-Bench",
            "scale": "H",
            "validation": "L",
            "diversity": "M",
            "eval_robust": "M",
            "repro": "M",
            "multi": "N",
            "overall_score": "18/30",
            "best_for": "Broad knowledge assessment",
            "concerns": "Validation provenance unclear, some synthetic"
        },
        {
            "benchmark": "SEvenLLM-Bench",
            "scale": "M",
            "validation": "N",
            "diversity": "M",
            "eval_robust": "M",
            "repro": "N",
            "multi": "M",
            "overall_score": "14/30",
            "best_for": "Multilingual evaluation",
            "concerns": "Validation undisclosed, not fully released"
        }
    ]

    with open(OUTPUT_DIR / "benchmark_recommendations.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=benchmarks[0].keys())
        writer.writeheader()
        writer.writerows(benchmarks)

    logger.info("✓ Benchmark recommendations CSV saved")


def generate_model_evaluation_checklist() -> None:
    """Generate checklist for model paper authors"""

    checklist = []
    checklist.append("# Cybersecurity LLM Model Evaluation Checklist")
    checklist.append("")
    checklist.append("Use this checklist when publishing results on cybersecurity LLM benchmarks.")
    checklist.append("")

    sections = [
        {
            "section": "Evaluation Harness",
            "items": [
                "Name the exact evaluation harness (lm-evaluation-harness v0.4.X? lighteval? custom?)",
                "Document evaluation backend (HuggingFace Transformers vs vLLM vs other)",
                "Provide pinned versions for all dependencies (transformers==X.Y.Z, vllm==A.B.C, etc.)",
                "Release code publicly; if not, explain why",
                "Test harness replicability with independent run"
            ]
        },
        {
            "section": "LLM Judge Evaluation",
            "items": [
                "If using LLM judge for structured extraction: report model, temperature, version",
                "Verify judge stability: run on identical responses ≥2 times, report consistency",
                "Document judge prompt templates (release in code)",
                "Explicitly state: NOT using bare regex for T-IDs, CWE-IDs, CVSS vectors",
                "Report any inference backend issues (NCCL deadlocks, truncations, etc.)"
            ]
        },
        {
            "section": "Benchmark Selection",
            "items": [
                "If using CTI-Bench: explicitly confirm you've fixed ATE evaluation (LLM judge, not regex)",
                "If using SECURE: exclude MAET/CWET/KCV from aggregate score or explain ceiling justification",
                "If comparing to RedSage: evaluate separately on RISys benchmarks vs independent benchmarks",
                "Disclose any author overlap with benchmark creators",
                "If treating ClusterName as one unit, state this explicitly"
            ]
        },
        {
            "section": "Reproducibility",
            "items": [
                "Report confidence intervals (95% CI) for all task accuracies (n ≥ 100 per task)",
                "Document random seeds, decoding parameters (temperature, top-p, etc.)",
                "Release evaluation code + example outputs + inference logs",
                "Include hardware details (GPU type, memory, CUDA version)",
                "Verify results with independent re-run on different hardware if possible"
            ]
        },
        {
            "section": "Multilingual Evaluation",
            "items": [
                "If claiming multilingual capability: evaluate on benchmark's non-English subset",
                "Report performance separately by language",
                "Disclose which languages are supported (don't conflate 'multilingual' with 2 languages)",
                "If not supporting Arabic: acknowledge the gap and explain why"
            ]
        },
        {
            "section": "Comparative Analysis",
            "items": [
                "Report results on independent benchmarks (CyberMetric, SecEval, CISSP) if available",
                "Include comparison to baseline models (Llama-3.1-8B-Instruct base)",
                "Show performance stratified by benchmark type (knowledge vs capability)",
                "If comparing to other models: use unified harness (not re-published results)"
            ]
        }
    ]

    for section in sections:
        checklist.append(f"## {section['section']}")
        checklist.append("")
        for i, item in enumerate(section['items'], 1):
            checklist.append(f"- [ ] {item}")
        checklist.append("")

    checklist.append("## Red Flags (Do Not Ignore)")
    checklist.append("")
    checklist.append("- ❌ Reporting ATE >0% F1 using original CTI-Bench regex evaluation")
    checklist.append("- ❌ SECURE scores where MAET/CWET/KCV are all 100%")
    checklist.append("- ❌ Results >15% higher on RISys benchmarks than independent benchmarks")
    checklist.append("- ❌ Using different harnesses across papers without normalization")
    checklist.append("- ❌ Confidence intervals with n < 50 per task")
    checklist.append("- ❌ Claiming 'multilingual' support without actual Arabic evaluation")
    checklist.append("")

    with open(OUTPUT_DIR / "model_eval_checklist.md", 'w') as f:
        f.write('\n'.join(checklist))

    logger.info("✓ Model evaluation checklist saved")


def generate_author_improvement_roadmap() -> None:
    """Generate roadmap for benchmark authors to improve"""

    roadmap = []
    roadmap.append("# Benchmark Author Improvement Roadmap")
    roadmap.append("")
    roadmap.append("Use this guide to improve your cybersecurity LLM benchmark.")
    roadmap.append("")

    benchmarks_roadmap = [
        {
            "benchmark": "CTI-Bench / AthenaBench",
            "urgency": "🔴 CRITICAL",
            "issues": [
                "ATE evaluation uses bare regex (detects format, not semantics)",
                "Validation provenance undisclosed (D2=N)",
                "Shared authors with RedSage model (undisclosed circular dependency)"
            ],
            "fixes": [
                "✅ Replace regex with LLM-as-Judge evaluation",
                "✅ Disclose validation sources (expert vs synthetic vs RAG)",
                "✅ Document author overlap with security LLM papers"
            ],
            "timeline": "0-1 month"
        },
        {
            "benchmark": "SECURE",
            "urgency": "🔴 CRITICAL",
            "issues": [
                "MAET, CWET, KCV tasks have 100% ceiling at base model level",
                "Task size undisclosed (D1=N)",
                "Shared authors with CTI-Bench/AthenaBench (undisclosed)"
            ],
            "fixes": [
                "✅ Replace ceiling tasks with harder variants from ICS advisories",
                "✅ Redesign MAET as open-ended extraction instead of 4-choice MCQ",
                "✅ Quantify and disclose size of task sets",
                "✅ Disclose author relationships with other benchmarks"
            ],
            "timeline": "1-3 months"
        },
        {
            "benchmark": "SecEval / CyberMetric / CISSP",
            "urgency": "🟡 MEDIUM",
            "issues": [
                "Limited task diversity (mostly MCQ)",
                "Multilingual coverage absent (D6=N)"
            ],
            "fixes": [
                "✅ Add open-ended extraction tasks (structured output, not MCQ)",
                "✅ Include Arabic subtask (or other non-English language)",
                "✅ Provide LLM-judge evaluation templates for reproducibility"
            ],
            "timeline": "2-6 months"
        },
        {
            "benchmark": "RedSage-Bench",
            "urgency": "🟡 MEDIUM",
            "issues": [
                "Validation provenance unclear (D2=L, 'partially synthetic')",
                "Multilingual coverage absent (D6=N)",
                "No code release for reproducibility (D5=M)"
            ],
            "fixes": [
                "✅ Quantify: how many items are expert vs synthetic vs RAG?",
                "✅ Add Arabic or another non-English language subset",
                "✅ Release evaluation code with pinned versions"
            ],
            "timeline": "3-6 months"
        }
    ]

    for bm in benchmarks_roadmap:
        roadmap.append(f"## {bm['benchmark']} {bm['urgency']}")
        roadmap.append("")
        roadmap.append("### Issues")
        for issue in bm['issues']:
            roadmap.append(f"- {issue}")
        roadmap.append("")
        roadmap.append("### Fixes")
        for fix in bm['fixes']:
            roadmap.append(f"- {fix}")
        roadmap.append("")
        roadmap.append(f"### Timeline: {bm['timeline']}")
        roadmap.append("")

    roadmap.append("## Universal Improvements (All Benchmarks)")
    roadmap.append("")
    roadmap.append("1. **Add LLM-Judge Evaluation Templates**")
    roadmap.append("   - Provide prompt templates for CTI tasks (RCM, RMS, ATE, TAA)")
    roadmap.append("   - Document reproducibility standards")
    roadmap.append("")
    roadmap.append("2. **Multilingual Expansion**")
    roadmap.append("   - Arabic is the highest priority (major threat actor language, UN official language)")
    roadmap.append("   - Chinese template already exists in SecBench")
    roadmap.append("")
    roadmap.append("3. **Transparent Validation**")
    roadmap.append("   - Document: what % expert-validated? synthetic? RAG-generated?")
    roadmap.append("   - Provide quality metrics (inter-rater agreement, validation coverage)")
    roadmap.append("")
    roadmap.append("4. **Backend Stability**")
    roadmap.append("   - Provide evaluation pipeline that works with both HF and vLLM")
    roadmap.append("   - Test and document compatibility by backend version")
    roadmap.append("")

    with open(OUTPUT_DIR / "author_improvement_roadmap.md", 'w') as f:
        f.write('\n'.join(roadmap))

    logger.info("✓ Author improvement roadmap saved")


def main():
    logger.info("=" * 70)
    logger.info("GENERATING COMPREHENSIVE REPORTS")
    logger.info("=" * 70)

    logger.info("Generating practitioner guide...")
    generate_practitioner_guide()

    logger.info("Generating benchmark recommendations...")
    generate_benchmark_recommendations_csv()

    logger.info("Generating model evaluation checklist...")
    generate_model_evaluation_checklist()

    logger.info("Generating author improvement roadmap...")
    generate_author_improvement_roadmap()

    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ All reports generated!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - practitioner_guide.md (for practitioners)")
    logger.info("  - benchmark_recommendations.csv (quality rankings)")
    logger.info("  - model_eval_checklist.md (for model paper authors)")
    logger.info("  - author_improvement_roadmap.md (for benchmark creators)")


if __name__ == "__main__":
    main()
