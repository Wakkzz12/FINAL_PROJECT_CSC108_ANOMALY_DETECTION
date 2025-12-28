def discretize_amount(amount, bins):
    if amount < bins[1]:
        return "LOW"
    elif amount < bins[2]:
        return "MEDIUM"
    return "HIGH"


def discretize_time(time):
    return "NORMAL" if time < 50000 else "UNUSUAL"


def discretize_risk(row):
    risk_score = abs(row["V1"]) + abs(row["V2"]) + abs(row["V3"])
    return "HIGH" if risk_score > 5 else "LOW"


def get_state(row, bins):
    return (
        discretize_amount(row["Amount"], bins),
        discretize_time(row["Time"]),
        discretize_risk(row)
    )
