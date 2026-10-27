# rules/rule_engine.py

# -------------------------------------------------------
# KNOWLEDGE BASE
# Each rule has:
#   id         : unique identifier
#   chain      : 'forward' or 'backward'
#   label      : short name
#   confidence : how strongly this rule applies (0.0–1.0)
#   condition  : function that takes facts → True/False
#   action     : advice text
# -------------------------------------------------------

RULES = [
    {
        'id':         'R1',
        'chain':      'forward',
        'label':      'Imminent Deadline',
        'confidence': 0.95,
        'condition':  lambda f: f['days_to_deadline'] <= 2,
        'action':     'Deadline is within 2 days. Drop all non-urgent tasks and focus entirely on this.'
    },
    {
        'id':         'R2',
        'chain':      'forward',
        'label':      'Deadline Within 3 Days',
        'confidence': 0.85,
        'condition':  lambda f: f['days_to_deadline'] <= 3,
        'action':     'Deadline approaching within 3 days. Set this as top priority immediately.'
    },
    {
        'id':         'R3',
        'chain':      'forward',
        'label':      'Low Available Time',
        'confidence': 0.80,
        'condition':  lambda f: f['avail_hours'] < 3,
        'action':     'Available study time is very low. Use 25-minute Pomodoro sessions to maximise focus.'
    },
    {
        'id':         'R4',
        'chain':      'forward',
        'label':      'High Stress',
        'confidence': 0.80,
        'condition':  lambda f: f['stress'] >= 7,
        'action':     'Stress level is high. Schedule a 30-minute break after every 2 hours of study.'
    },
    {
        'id':         'R5',
        'chain':      'forward',
        'label':      'Sleep Deprivation',
        'confidence': 0.90,
        'condition':  lambda f: f['sleep'] < 6,
        'action':     'Sleeping less than 6 hours severely reduces cognitive performance. Prioritise sleep tonight.'
    },
    {
        'id':         'R6',
        'chain':      'forward',
        'label':      'Low Confidence',
        'confidence': 0.75,
        'condition':  lambda f: f['confidence'] < 40,
        'action':     'Confidence is below 40%. Break tasks into smaller steps and track each small win.'
    },
    {
        'id':         'R7',
        'chain':      'forward',
        'label':      'Heavy Subject Load',
        'confidence': 0.70,
        'condition':  lambda f: f['subjects'] >= 6,
        'action':     'Managing 6+ subjects. Create a priority matrix and focus on highest-weight assessments first.'
    },
    {
        'id':         'R8',
        'chain':      'forward',
        'label':      'Confidence-Stress Mismatch',
        'confidence': 0.90,
        'condition':  lambda f: f['confidence'] < 40 and f['stress'] >= 7,
        'action':     'Critical: low confidence AND high stress detected. High burnout risk — contact student support today.'
    },
    {
        'id':         'R9',
        'chain':      'forward',
        'label':      'Missed Sessions Risk',
        'confidence': 0.70,
        'condition':  lambda f: f['avail_hours'] * 7 < f['hours_needed'] * 0.6,
        'action':     'You risk missing study sessions. Consider negotiating a deadline extension.'
    },
    {
        'id':         'R10',
        'chain':      'forward',
        'label':      'Collaborative Study Recommended',
        'confidence': 0.65,
        'condition':  lambda f: f['gender'] in ['female', 'non-binary'],
        'action':     'Research supports collaborative study for this group. Consider joining a study group.'
    },
    {
        'id':         'R11',
        'chain':      'forward',
        'label':      'High Workload Warning',
        'confidence': 0.85,
        'condition':  lambda f: f['hours_needed'] > 25,
        'action':     'Weekly study load exceeds 25 hours. Distribute evenly — avoid cramming on one day.'
    },
    {
        'id':         'R12',
        'chain':      'backward',
        'label':      'High Risk — Root Cause Analysis',
        'confidence': 0.90,
        'condition':  lambda f: f['risk_score'] > 65,
        'action':     'GOAL: Reduce risk. Tracing causes → check deadline urgency (R1,R2), stress (R4), sleep (R5).'
    },
    {
        'id':         'R13',
        'chain':      'backward',
        'label':      'Capacity Overload',
        'confidence': 0.85,
        'condition':  lambda f: f['hours_needed'] > f['avail_hours'] * 7,
        'action':     'GOAL: Fit workload into available time. Root cause → too many subjects or too little free time.'
    },
    {
        'id':         'R14',
        'chain':      'forward',
        'label':      'Missed Sessions Warning',
        'confidence': 0.80,
        'condition':  lambda f: f.get('missed_sessions', 0) >= 2 and f.get('confidence', 100) < 50,
        'action':     'Missed 2+ sessions with low confidence — risk increases. Catch up on missed material immediately.'
    },
    {
        'id':         'R15',
        'chain':      'forward',
        'label':      'Good Academic Standing',
        'confidence': 0.90,
        'condition':  lambda f: f.get('risk_score', 100) < 30,
        'action':     'Risk is LOW. Maintain current study habits, stay consistent, and use spare time for revision.'
    },
]

# -------------------------------------------------------
# FORWARD CHAINING
# Facts → check all forward rules → collect fired rules
# -------------------------------------------------------
def forward_chain(facts):
    fired = []
    for rule in RULES:
        if rule['chain'] == 'forward':
            if rule['condition'](facts):
                fired.append(rule)
    return fired

# -------------------------------------------------------
# BACKWARD CHAINING
# Goal → find supporting rules → trace root causes
# Also decides which follow-up question to ask next
# -------------------------------------------------------
def backward_chain(facts):
    fired_backward = []
    trace          = []

    for rule in RULES:
        if rule['chain'] == 'backward':
            if rule['condition'](facts):
                fired_backward.append(rule)
                trace.append(
                    f"[{rule['id']}] {rule['label']} "
                    f"(confidence {rule['confidence']}) → confirmed"
                )

    # Trace root causes from forward rules
    root_cause_ids = ['R1', 'R2', 'R4', 'R5', 'R8']
    for rule in RULES:
        if rule['id'] in root_cause_ids:
            if rule['condition'](facts):
                trace.append(
                    f"  ↳ Root cause [{rule['id']}] "
                    f"{rule['label']} (conf: {rule['confidence']})"
                )

    return {
        'goal':  'Reduce student risk level',
        'fired': fired_backward,
        'trace': trace
    }

# -------------------------------------------------------
# BACKWARD CHAINING → drives follow-up questions
# Returns the most important question to ask next
# based on which facts are uncertain or missing
# -------------------------------------------------------
def get_followup_question(facts):
    """
    Uses backward chaining logic to decide which
    follow-up question is most important to ask next.
    """
    # Goal: explain high stress → ask cause
    if facts.get('stress', 0) >= 7 and 'stress_cause' not in facts:
        return {
            'reason': 'R4 fired — stress is high but cause unknown',
            'question': 'What is causing your high stress? (exam / workload / personal / other)'
        }

    # Goal: explain low confidence → ask hardest subject
    if facts.get('confidence', 100) < 40 and 'hardest_subject' not in facts:
        return {
            'reason': 'R6 fired — confidence low but weak subject unknown',
            'question': 'Which subject are you least confident in?'
        }

    # Goal: resolve capacity overload → ask about extensions
    capacity = facts.get('avail_hours', 4) * 7
    if facts.get('hours_needed', 0) > capacity and 'extension_possible' not in facts:
        return {
            'reason': 'R13 fired — workload exceeds capacity',
            'question': 'Can you request a deadline extension from your lecturer? (yes / no)'
        }

    # Goal: imminent deadline → ask if plan exists
    if facts.get('days_to_deadline', 7) <= 2 and 'has_plan' not in facts:
        return {
            'reason': 'R1 fired — deadline imminent',
            'question': 'Do you already have a study plan for this deadline? (yes / no)'
        }

    # Default fallback — always ask at least one question
    return {
        'reason': 'General check-in',
        'question': 'How would you rate your current study environment? (good / ok / poor)'
    }


# -------------------------------------------------------
# TEST
# -------------------------------------------------------
if __name__ == "__main__":
    test_facts = {
        'subjects':          7,
        'deadlines':         4,
        'days_to_deadline':  2,
        'hours_needed':      30,
        'avail_hours':       2.0,
        'confidence':        20,
        'stress':            9,
        'sleep':             5.0,
        'gender':            'female',
        'risk_score':        85
    }

    print("=== FORWARD CHAINING ===")
    fired = forward_chain(test_facts)
    for r in fired:
        print(f"  [{r['id']}] {r['label']} "
              f"(conf: {r['confidence']}) → {r['action']}")

    print("\n=== BACKWARD CHAINING ===")
    result = backward_chain(test_facts)
    print(f"  Goal: {result['goal']}")
    for line in result['trace']:
        print(f"  {line}")

    print("\n=== FOLLOW-UP QUESTION (driven by backward chaining) ===")
    fq = get_followup_question(test_facts)
    print(f"  Reason   : {fq['reason']}")
    print(f"  Question : {fq['question']}")