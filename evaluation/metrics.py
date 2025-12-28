def detection_rate(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def false_positive_rate(fp, tn):
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0