import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    total = len(labels)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in Counter(labels).values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    total = len(examples)
    if total == 0:
        return 0.0
    base_entropy = calculate_entropy([ex[target_attr] for ex in examples])
    subsets = {}
    for ex in examples:
        subsets.setdefault(ex[attr], []).append(ex[target_attr])

    weighted_entropy = sum(
        (len(labels) / total) * calculate_entropy(labels)
        for labels in subsets.values()
    )
    return base_entropy - weighted_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    max_count = max(counts.values())
    return min(label for label, c in counts.items() if c == max_count)

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    if not examples:
        return {}

    labels = [ex[target_attr] for ex in examples]

    if len(set(labels)) == 1:
        return labels[0]

    if not attributes:
        return majority_class(examples, target_attr)

    best_attr = None
    best_gain = -1.0
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    tree = {best_attr: {}}
    remaining = [a for a in attributes if a != best_attr]

    for value in sorted(set(ex[best_attr] for ex in examples)):
        subset = [ex for ex in examples if ex[best_attr] == value]
        tree[best_attr][value] = learn_decision_tree(subset, remaining, target_attr)

    return tree