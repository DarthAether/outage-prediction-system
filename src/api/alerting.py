def classify_risk(probability: float):
    """
    Converts outage risk probability into alert levels.
    """
    if probability >= 0.85:
        return "RED"
    elif probability >= 0.60:
        return "ORANGE"
    elif probability >= 0.40:
        return "YELLOW"
    else:
        return "GREEN"


def response_actions(risk_level: str):
    """
    Maps risk levels to disaster-response actions.
    """
    actions = {
        "GREEN": [
            "Normal grid operation",
            "Routine monitoring"
        ],
        "YELLOW": [
            "Increase grid monitoring",
            "Prepare emergency response teams",
            "Notify local utilities"
        ],
        "ORANGE": [
            "Pre-position repair crews",
            "Alert hospitals and emergency shelters",
            "Secure critical infrastructure"
        ],
        "RED": [
            "Activate disaster response forces",
            "Deploy emergency medical teams",
            "Isolate high-risk power lines",
            "Prioritize power restoration planning"
        ]
    }
    return actions[risk_level]
