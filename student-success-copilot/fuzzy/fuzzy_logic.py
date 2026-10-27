# fuzzy/fuzzy_logic.py

# -------------------------------------------------------
# FUZZY LOGIC CONTROLLER
# Converts crisp student inputs into fuzzy sets,
# applies fuzzy rules, and outputs recommended
# study intensity level for the week.
# -------------------------------------------------------

# -------------------------------------------------------
# MEMBERSHIP FUNCTIONS
# Each function returns a value between 0.0 and 1.0
# representing how strongly the input belongs to that set
# -------------------------------------------------------

def trimf(x, a, b, c):
    """
    Triangular membership function.
    a = left foot, b = peak, c = right foot
    Returns membership value 0.0–1.0
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def trapmf(x, a, b, c, d):
    """
    Trapezoidal membership function.
    a = left foot, b = left shoulder,
    c = right shoulder, d = right foot
    Returns membership value 0.0–1.0
    """
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (d - x) / (d - c)

# -------------------------------------------------------
# FUZZIFICATION
# Convert crisp numbers → fuzzy membership degrees
# -------------------------------------------------------

def fuzzify_stress(stress):
    """
    Stress: 1–10
    LOW    : 1–4
    MEDIUM : 3–7
    HIGH   : 6–10
    """
    return {
        'LOW':    trapmf(stress, 1, 1, 3, 5),
        'MEDIUM': trimf(stress,  3, 5, 7),
        'HIGH':   trapmf(stress, 5, 7, 10, 10)
    }

def fuzzify_workload(avail_hours):
    """
    Workload based on available hours per day (inverse — less time = heavier workload)
    avail_hours: 1.5–8.0
    HEAVY    : 1.5–3.5  (little time = heavy workload)
    MODERATE : 3.0–6.0
    LIGHT    : 5.5–8.0
    """
    return {
        'HEAVY':    trapmf(avail_hours, 1.5, 1.5, 2.5, 4.0),
        'MODERATE': trimf(avail_hours,  2.5, 4.5, 6.5),
        'LIGHT':    trapmf(avail_hours, 5.0, 6.5, 8.0, 8.0)
    }

def fuzzify_confidence(confidence):
    """
    Confidence: 0–100%
    LOW    : 0–40
    MEDIUM : 30–70
    HIGH   : 60–100
    """
    return {
        'LOW':    trapmf(confidence, 0,  0,  25, 45),
        'MEDIUM': trimf(confidence,  30, 50, 70),
        'HIGH':   trapmf(confidence, 55, 75, 100, 100)
    }

# -------------------------------------------------------
# FUZZY RULES
# Each rule returns (intensity_label, firing_strength)
# firing_strength = min of all input memberships (AND)
# -------------------------------------------------------

def apply_rules(stress_sets, workload_sets, confidence_sets):
    """
    Fuzzy rule base — 12 rules
    Output fuzzy sets for intensity:
      LIGHT / MODERATE / INTENSE / CRITICAL
    """
    rules = []

    s = stress_sets
    w = workload_sets
    c = confidence_sets

    # --- CRITICAL intensity rules ---
    # IF stress HIGH AND workload HEAVY AND confidence LOW → CRITICAL
    rules.append(('CRITICAL',
        min(s['HIGH'], w['HEAVY'], c['LOW'])))

    # IF stress HIGH AND workload HEAVY → CRITICAL
    rules.append(('CRITICAL',
        min(s['HIGH'], w['HEAVY'])))

    # --- INTENSE intensity rules ---
    # IF stress HIGH AND workload MODERATE → INTENSE
    rules.append(('INTENSE',
        min(s['HIGH'], w['MODERATE'])))

    # IF stress MEDIUM AND workload HEAVY → INTENSE
    rules.append(('INTENSE',
        min(s['MEDIUM'], w['HEAVY'])))

    # IF stress HIGH AND confidence LOW → INTENSE
    rules.append(('INTENSE',
        min(s['HIGH'], c['LOW'])))

    # IF stress MEDIUM AND confidence LOW AND workload MODERATE → INTENSE
    rules.append(('INTENSE',
        min(s['MEDIUM'], c['LOW'], w['MODERATE'])))

    # --- MODERATE intensity rules ---
    # IF stress MEDIUM AND workload MODERATE → MODERATE
    rules.append(('MODERATE',
        min(s['MEDIUM'], w['MODERATE'])))

    # IF stress LOW AND workload HEAVY → MODERATE
    rules.append(('MODERATE',
        min(s['LOW'], w['HEAVY'])))

    # IF stress MEDIUM AND confidence MEDIUM → MODERATE
    rules.append(('MODERATE',
        min(s['MEDIUM'], c['MEDIUM'])))

    # IF stress LOW AND confidence MEDIUM → MODERATE
    rules.append(('MODERATE',
        min(s['LOW'], c['MEDIUM'])))

    # --- LIGHT intensity rules ---
    # IF stress LOW AND workload LIGHT → LIGHT
    rules.append(('LIGHT',
        min(s['LOW'], w['LIGHT'])))

    # IF stress LOW AND confidence HIGH → LIGHT
    rules.append(('LIGHT',
        min(s['LOW'], c['HIGH'])))

    return rules

# -------------------------------------------------------
# DEFUZZIFICATION
# Aggregate fired rules → crisp intensity score
# Using weighted average (centre of gravity simplified)
# -------------------------------------------------------

def defuzzify(rules):
    """
    Convert fuzzy output to crisp intensity score 0–100
    LIGHT=25, MODERATE=50, INTENSE=75, CRITICAL=100
    """
    label_values = {
        'LIGHT':    25,
        'MODERATE': 50,
        'INTENSE':  75,
        'CRITICAL': 100
    }

    numerator   = 0.0
    denominator = 0.0

    for label, strength in rules:
        if strength > 0:
            value        = label_values[label]
            numerator   += strength * value
            denominator += strength

    if denominator == 0:
        return 50.0  # default moderate

    return round(numerator / denominator, 1)

# -------------------------------------------------------
# INTENSITY LABEL + ADJUSTMENT MESSAGE
# -------------------------------------------------------

def get_intensity_label(score):
    if score >= 80:
        return 'CRITICAL', '🔴 Maximum effort needed. Study every available hour this week.'
    elif score >= 60:
        return 'INTENSE',  '🟠 High intensity week. Minimise distractions and stick to the plan.'
    elif score >= 35:
        return 'MODERATE', '🟡 Moderate intensity. Maintain steady progress with planned breaks.'
    else:
        return 'LIGHT',    '🟢 Light intensity. Stay consistent but avoid overworking.'

# -------------------------------------------------------
# MAIN CONTROLLER FUNCTION
# -------------------------------------------------------

def fuzzy_controller(stress, avail_hours, confidence):
    """
    Main entry point for fuzzy logic controller.

    Inputs:
      stress      : int 1–10
      avail_hours : float 1.5–8.0
      confidence  : int 0–100

    Returns dict with:
      intensity_score  : 0–100
      intensity_label  : LIGHT/MODERATE/INTENSE/CRITICAL
      adjustment       : advice string
      memberships      : all fuzzy membership values (for report)
    """
    # Step 1: Fuzzify inputs
    stress_sets     = fuzzify_stress(stress)
    workload_sets   = fuzzify_workload(avail_hours)
    confidence_sets = fuzzify_confidence(confidence)

    # Step 2: Apply rules
    fired_rules = apply_rules(stress_sets, workload_sets, confidence_sets)

    # Step 3: Defuzzify
    intensity_score = defuzzify(fired_rules)

    # Step 4: Get label and advice
    intensity_label, adjustment = get_intensity_label(intensity_score)

    return {
        'intensity_score': intensity_score,
        'intensity_label': intensity_label,
        'adjustment':      adjustment,
        'memberships': {
            'stress':     stress_sets,
            'workload':   workload_sets,
            'confidence': confidence_sets
        }
    }

# -------------------------------------------------------
# TEST
# -------------------------------------------------------
if __name__ == "__main__":
    print("=== FUZZY LOGIC CONTROLLER TEST ===\n")

    test_cases = [
        {'name': 'Alex (Struggling)',      'stress': 9, 'avail': 2.0, 'conf': 20},
        {'name': 'Jordan (Well-Prepared)', 'stress': 2, 'avail': 6.0, 'conf': 85},
        {'name': 'Sam (Borderline)',       'stress': 5, 'avail': 4.0, 'conf': 55},
    ]

    for tc in test_cases:
        result = fuzzy_controller(tc['stress'], tc['avail'], tc['conf'])
        print(f"  Student  : {tc['name']}")
        print(f"  Inputs   : stress={tc['stress']}, "
              f"avail={tc['avail']}h, conf={tc['conf']}%")
        print(f"  Score    : {result['intensity_score']}/100")
        print(f"  Intensity: {result['intensity_label']}")
        print(f"  Advice   : {result['adjustment']}")
        print(f"  Memberships:")
        for key, sets in result['memberships'].items():
            vals = ", ".join(f"{k}={v:.2f}" for k,v in sets.items())
            print(f"    {key:<12}: {vals}")
        print()