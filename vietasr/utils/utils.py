import editdistance


def calculate_wer(preds: str, targets: str, use_cer=False) -> float:
    """Calculate sentence-level WER score.

    Args:
        pred: Prediction character sequences. (B, ?)
        target: Target character sequences. (B, ?)
        use_cer: bool

    Returns:
        : Average WER score

    """

    distances, lens = [], []

    for pred, target in zip(preds, targets):
        if use_cer:
            pred = list(pred)
            target = list(target)
        else:
            pred = pred.split()
            target = target.split()

        distances.append(editdistance.eval(pred, target))
        lens.append(len(target))

    return float(sum(distances)) / sum(lens) * 100
