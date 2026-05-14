#!/usr/bin/env python3
import json
from pathlib import Path

def regenerate_summary(detailed_dir: str):
    detailed_path = Path(detailed_dir)
    tasks = {}
    total_correct = 0
    total_samples = 0

    for f in sorted(detailed_path.glob('*_detailed.jsonl')):
        task = f.stem.replace('_detailed', '').upper()
        objects = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        objects.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not objects:
            continue

        total = len(objects)

        if 'mad' in objects[0]:
            mad_scores = [o.get('mad', 10.0) for o in objects]
            mean_mad = sum(mad_scores) / len(mad_scores)
            acc = max(0.0, 1.0 - mean_mad / 7.7)
            pseudo_correct = round(acc * total)
            tasks[task] = {'accuracy': round(acc, 4), 'correct': pseudo_correct,
                           'total': total, 'mean_mad': round(mean_mad, 3)}
            total_correct += pseudo_correct
        elif 'tp' in objects[0]:
            tp = sum(o.get('tp', 0) for o in objects)
            fp = sum(o.get('fp', 0) for o in objects)
            fn = sum(o.get('fn', 0) for o in objects)
            exact = sum(1 for o in objects if o.get('exact_match'))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            tasks[task] = {'accuracy': round(f1, 4), 'correct': exact, 'total': total,
                           'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)}
            total_correct += exact
        else:
            correct = sum(1 for o in objects if o.get('is_correct'))
            acc = correct / total if total > 0 else 0
            tasks[task] = {'accuracy': round(acc, 4), 'correct': correct, 'total': total}
            total_correct += correct

        total_samples += total

    summary = {
        'tasks': tasks,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'overall_accuracy': round(total_correct / total_samples, 4) if total_samples else 0,
    }
    out = detailed_path / 'summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

models = [
    'GPT-5.4',
    'Fanar-2-27B-Instruct',
    'Foundation-Sec-8B-Instruct',
    'Gemma-4-31B-it',
    'GPT-oss-20B',
    'Llama-Primus-Merged',
    'Qwen3.6-35B-A3B',
    'RedSage-Qwen3-8B-DPO',
]

print(f"{'Model':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
print('-' * 62)
for model in models:
    d = f'outputs/judge_{model}/eval_results'
    if not Path(d).is_dir():
        print(f'{model:<30} NO DATA')
        continue
    s = regenerate_summary(d)
    t = s['total_samples']
    c = s['total_correct']
    a = s['overall_accuracy']
    print(f'{model:<30} {a:>9.1%} {c:>10} {t:>8}')
