def discretize_amount(amount, bins):
    # Discretize transaction amount into LOW, MEDIUM, HIGH based on bins
    if amount < bins[1]:
        return "LOW"
    elif amount < bins[2]:
        return "MEDIUM"
    return "HIGH"


def discretize_time(time):
    # Discretize time into NORMAL and UNUSUAL
    return "NORMAL" if time < 50000 else "UNUSUAL"


def discretize_risk(row):
    # Simple risk discretization based on V1, V2, V3 features
    risk_score = abs(row["V1"]) + abs(row["V2"]) + abs(row["V3"])
    return "HIGH" if risk_score > 5 else "LOW"


def get_state(row, bins):
    # Get discretized state tuple (amount, time, risk)
    return (
        discretize_amount(row["Amount"], bins),
        discretize_time(row["Time"]),
        discretize_risk(row)
    )
