"""Shared metric computation for evaluation scripts."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/extra whitespace. (HELMET-compatible)"""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def substring_match(pred: str, gold: str) -> bool:
    return normalize_answer(gold) in normalize_answer(pred)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_over_answers(metric_fn, prediction: str, answers: list[str]):
    """Compute max metric score over all ground truth answers."""
    if isinstance(answers, str):
        answers = [answers]
    elif answers and isinstance(answers[0], list):
        answers = [a for sublist in answers for a in sublist]
    return max(metric_fn(prediction, gt) for gt in answers)


def aggregate(results: list[dict], keys: list[str]) -> dict:
    """Average metric values across results."""
    if not results:
        return {}
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
