from typing import List, Tuple

from models.fact import Fact


def calc_ap(pred_fact_conf_tuples: List[Tuple[Fact, float]], gt_facts: List[Fact]) -> float:
    ap = 0

    pred_fact_conf_tuples.sort(key=lambda x: x[1], reverse=True)
    sorted_pred_facts = [fact for fact, _ in pred_fact_conf_tuples]

    correct = 0
    for i, fact in enumerate(sorted_pred_facts):
        if fact in gt_facts:
            correct += 1
            ap += correct / (i + 1)

    return (ap / len(gt_facts)) if len(gt_facts) > 0 else 1
