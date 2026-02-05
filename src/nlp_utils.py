# Keywords that definitely indicate a Major Accident
HIGH_RISK_KEYWORDS = [
    'rolled over', 'rollover', 'head-on', 'head on',
    'fire', 'caught fire',
    'pedestrian', 'unconscious',
    'trapped', 'fatal', 'ambulance', 'totaled'
]

def check_risk_keywords(text):
    """Returns 1 if text contains high-risk keywords, else 0."""
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    for word in HIGH_RISK_KEYWORDS:
        if word in text_lower:
            return 1
    return 0
