# ui/app.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.search_planner import build_tasks, greedy_planner, astar_planner
from rules.rule_engine import forward_chain, backward_chain, get_followup_question
from ml.ml_model import predict
from fuzzy.fuzzy_logic import fuzzy_controller

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def line(char='─', n=58):
    print(char * n)

def header(title):
    line()
    print(f"  {title}")
    line()

# -------------------------------------------------------
# INPUT COLLECTION
# -------------------------------------------------------
def get_inputs():
    clear()
    header("🎓 STUDENT SUCCESS COPILOT")
    print("  Please answer the following questions.\n")

    name = input("  Your name: ").strip() or "Student"

    print("\n  Gender (male / female / non-binary / prefer not to say):")
    gender = input("  → ").strip().lower() or "prefer not to say"

    print("\n  Year of study (1 / 2 / 3 / 4 / pg):")
    year = input("  → ").strip() or "1"

    print("\n  How many active subjects do you have?")
    subjects = int(input("  → ") or 4)

    print(f"\n  Enter your {subjects} subject names one by one:")
    subject_names = []
    for i in range(subjects):
        sname = input(f"    Subject {i+1}: ").strip()
        if not sname:
            sname = f"Subject {i+1}"
        subject_names.append(sname)

    print("\n  Number of deadlines in the next 7 days:")
    while True:
        deadlines = int(input("  → ") or 0)
        if deadlines <= subjects:
            break
        print(f"  ⚠ Cannot have more deadlines ({deadlines}) "
              f"than subjects ({subjects}). Please re-enter.")

    print("\n  Days until your nearest deadline:")
    days_to_deadline = int(input("  → ") or 7)

    print("\n  Total study hours needed this week:")
    hours_needed = int(input("  → ") or 15)

    print("\n  Available study hours per day:")
    avail_hours = float(input("  → ") or 4)

    print("\n  Confidence level (0-100%):")
    confidence = int(input("  → ") or 50)

    print("\n  Stress level (1=calm, 10=very stressed):")
    stress = int(input("  → ") or 5)

    print("\n  Average sleep hours per night:")
    sleep = float(input("  → ") or 7)

    print("\n  Total study sessions planned this week:")
    total_sessions = int(input("  → ") or 10)

    print("\n  How many of those sessions have you already missed?")
    missed_sessions = int(input("  → ") or 0)

    return {
        'name':             name,
        'gender':           gender,
        'year':             year,
        'subjects':         subjects,
        'subject_names':    subject_names,
        'deadlines':        deadlines,
        'days_to_deadline': days_to_deadline,
        'hours_needed':     hours_needed,
        'avail_hours':      avail_hours,
        'confidence':       confidence,
        'stress':           stress,
        'sleep':            sleep,
        'total_sessions':   total_sessions,
        'missed_sessions':  missed_sessions
    }

# -------------------------------------------------------
# FOLLOW-UP QUESTIONS — driven by backward chaining
# -------------------------------------------------------
def ask_followups(data):
    questions_asked = 0

    line('·')
    print("\n  FOLLOW-UP QUESTIONS (driven by backward chaining)\n")

    # Question 1 — driven by backward chaining
    fq = get_followup_question(data)
    print(f"  Triggered by: {fq['reason']}")
    answer = input(f"  ❓ {fq['question']}\n  → ").strip()

    if 'stress' in fq['reason']:
        data['stress_cause'] = answer or 'unspecified'
    elif 'confidence' in fq['reason']:
        data['hardest_subject'] = answer or 'unspecified'
    elif 'capacity' in fq['reason']:
        data['extension_possible'] = answer.lower() == 'yes'
    elif 'deadline' in fq['reason']:
        data['has_plan'] = answer.lower() == 'yes'
    else:
        data['environment'] = answer or 'ok'
    questions_asked += 1

    # Question 2 — detect contradictory information
    contradiction = _detect_contradiction(data)
    if contradiction:
        print(f"\n  Triggered by: {contradiction['reason']}")
        answer2 = input(f"  ❓ {contradiction['question']}\n  → ").strip()
        data[contradiction['key']] = answer2 or 'unspecified'
        questions_asked += 1

    # Always ask at least one general check-in if only 1 asked
    if questions_asked < 2:
        print(f"\n  Triggered by: General check-in")
        answer3 = input(
            f"  ❓ How would you rate your study environment? "
            f"(good / ok / poor)\n  → "
        ).strip()
        data['environment'] = answer3 or 'ok'

    return data


def _detect_contradiction(data):
    """
    Detects contradictory or inconsistent inputs and
    returns a follow-up question to resolve them.
    """
    # Contradiction: high confidence but also high stress
    if data['confidence'] > 70 and data['stress'] >= 7:
        return {
            'reason':   'Contradiction — high confidence but high stress',
            'question': 'You seem confident but also stressed. '
                        'What is the main source of your stress?',
            'key':      'stress_cause'
        }

    # Contradiction: many missed sessions but low stress
    missed_pct = (data['missed_sessions'] /
                  max(data['total_sessions'], 1)) * 100
    if missed_pct >= 40 and data['stress'] < 4:
        return {
            'reason':   'Contradiction — many missed sessions but low stress',
            'question': f"You missed {data['missed_sessions']} of "
                        f"{data['total_sessions']} sessions but seem calm. "
                        f"Are you aware of the backlog? (yes / no)",
            'key':      'backlog_aware'
        }

    # Contradiction: very few hours available but many deadlines
    if data['avail_hours'] < 2 and data['deadlines'] >= 3:
        return {
            'reason':   'Contradiction — very limited time with many deadlines',
            'question': 'You have many deadlines but very little time. '
                        'Can you reduce other commitments this week? (yes / no)',
            'key':      'can_reduce_commitments'
        }

    return None

# -------------------------------------------------------
# OVERALL EXPLANATION
# -------------------------------------------------------
def generate_explanation(data, ml_result, fired_rules, fuzzy_result):
    """
    Generates overall plain-English explanation of
    why this plan and risk level were assigned.
    """
    risk    = ml_result['risk_score']
    label   = ml_result['risk_label']
    top3    = fired_rules[:3] if fired_rules else []
    reasons = ', '.join(r['label'] for r in top3) if top3 else \
              'general workload analysis'

    missed_pct = round(
        (data.get('missed_sessions', 0) /
         max(data.get('total_sessions', 1), 1)) * 100
    )

    line('·')
    print(f"\n  WHY THIS PLAN\n")
    print(f"  Risk is {label} ({risk}/100) due to: {reasons}.")
    print(f"  Intensity is {fuzzy_result['intensity_label']} "
          f"based on stress, workload and confidence.")
    print(f"  A* schedule prioritises your "
          f"{data['deadlines']} deadline(s) first.")

    if data.get('missed_sessions', 0) > 0:
        print(f"  You have missed "
              f"{data['missed_sessions']}/"
              f"{data['total_sessions']} sessions "
              f"({missed_pct}%) — recovery time is built into the plan.")

    if data.get('stress_cause'):
        print(f"  Stress cause noted: {data['stress_cause']}.")
    if data.get('environment'):
        print(f"  Study environment: {data['environment']}.")
    if data.get('hardest_subject'):
        print(f"  Focus extra time on: {data['hardest_subject']}.")

# -------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------
def show_results(data, ml_result, fuzzy_result,
                 fired_rules, bwd_result, greedy, astar):
    clear()
    header(f"📊 RESULTS — {data['name'].upper()}")

    # --- ML Prediction ---
    risk  = ml_result['risk_score']
    label = ml_result['risk_label']
    rec   = ml_result['rec_hours']

    icons = {
        'LOW':      '🟢',
        'MODERATE': '🟡',
        'HIGH':     '🟠',
        'CRITICAL': '🔴'
    }
    icon = icons.get(label, '⚪')

    print(f"\n  {icon} RISK LEVEL        : {label} ({risk}/100)")
    print(f"  📚 REC. HOURS       : {rec}h this week")
    print(f"  📅 NEAREST DEADLINE : in {data['days_to_deadline']} day(s)")
    print(f"  🤖 ML Models        : RandomForest (risk) + "
          f"LinearRegression (hours)")
    print(f"\n  ⚠  ML Limitations   : Trained on synthetic data (200 rows).")
    print(f"     Metrics          : Risk R²=0.929, MAE=5.88 | "
          f"Label Acc=97.5%, F1=0.98")
    print(f"                        Hours R²=1.00 "
          f"(derived formula — not generalisable).")
    print(f"     Note             : Real student data may produce "
          f"different predictions.")

    # --- Overall Explanation ---
    generate_explanation(data, ml_result, fired_rules, fuzzy_result)

    # --- Fuzzy Logic ---
    line('·')
    print(f"\n  FUZZY LOGIC CONTROLLER\n")
    print(f"  Intensity Score : {fuzzy_result['intensity_score']}/100")
    print(f"  Intensity Level : {fuzzy_result['intensity_label']}")
    print(f"  Adjustment      : {fuzzy_result['adjustment']}")
    print(f"\n  Membership Values:")
    for key, sets in fuzzy_result['memberships'].items():
        vals = "  |  ".join(
            f"{k}={v:.2f}" for k, v in sets.items()
        )
        print(f"    {key:<12}: {vals}")

    # --- Rule Engine ---
    line('·')
    print(f"\n  RULE ENGINE — {len(fired_rules)} rule(s) fired\n")
    if fired_rules:
        for r in fired_rules:
            print(f"  [{r['id']}] {r['label']} "
                  f"[confidence: {r['confidence']}]")
            print(f"       → {r['action']}\n")
    else:
        print("  No rules fired — student profile looks balanced.\n")

    # --- Backward Chaining ---
    line('·')
    print(f"\n  BACKWARD CHAINING — Goal: {bwd_result['goal']}\n")
    if bwd_result['trace']:
        for t in bwd_result['trace']:
            print(f"  {t}")
    else:
        print("  No high-risk goal triggered — student is on track.")

    # --- Search Comparison ---
    line('·')
    print(f"\n  SEARCH PLANNER COMPARISON\n")
    print(f"  {'Method':<22} {'Nodes':>6} "
          f"{'Backtracks':>10} {'Time(ms)':>10}")
    line('-', 55)
    print(f"  {greedy['method']:<22} "
          f"{greedy['nodes']:>6} "
          f"{greedy['backtracks']:>10} "
          f"{greedy['time_ms']:>10}")
    print(f"  {astar['method']:<22} "
          f"{astar['nodes']:>6} "
          f"{astar['backtracks']:>10} "
          f"{astar['time_ms']:>10}")
    print(f"\n  → A* selected: balances urgency + effort estimate")
    print(f"  → Quality: A* h=hours/deadline_day gives smarter ordering.")
    print(f"  → Greedy: faster but purely local — "
          f"may miss globally better orderings.")

    # --- Search Problem Formulation ---
    line('·')
    print(f"\n  SEARCH PROBLEM FORMULATION\n")
    names = ', '.join(
        data.get('subject_names',
                 [f"S{i+1}" for i in range(data['subjects'])])
    )
    print(f"  Subjects: {names}")
    print(f"  State   : {data['hours_needed']}h needed | "
          f"7 days available")
    print(f"  Actions : allocate up to 2.5h per subject per day")
    print(f"  Goal    : all tasks scheduled before their deadlines")
    print(f"  Cost    : nodes explored = {astar['nodes']} | "
          f"backtracks = {astar['backtracks']}")

    # --- Weekly Schedule ---
    line('·')
    print(f"\n  WEEKLY STUDY PLAN — A* Heuristic\n")
    for day, slots in astar['schedule'].items():
        if slots:
            items = ", ".join(
                f"{s['task']} {s['hours']}h" for s in slots
            )
            print(f"  {day:<12}: {items}")
        else:
            print(f"  {day:<12}: Rest / buffer day")

    line()
    print("\n  Press Enter to return to menu...")
    input()

# -------------------------------------------------------
# CORE ANALYSIS
# -------------------------------------------------------
def _run_analysis(data):
    # ML prediction
    ml_result = predict(
        subjects         = data['subjects'],
        deadlines        = data['deadlines'],
        days_to_deadline = data['days_to_deadline'],
        hours_needed     = data['hours_needed'],
        avail_hours      = data['avail_hours'],
        confidence       = data['confidence'],
        stress           = data['stress'],
        sleep            = data['sleep'],
        gender           = data['gender']
    )

    # Fuzzy logic controller
    fuzzy_result = fuzzy_controller(
        stress      = data['stress'],
        avail_hours = data['avail_hours'],
        confidence  = data['confidence']
    )

    # Add risk score and session data to facts
    facts = {
        **data,
        'risk_score': ml_result['risk_score']
    }

    # Rule engine
    fired_rules = forward_chain(facts)
    bwd_result  = backward_chain(facts)

    # Search planner
    tasks  = build_tasks(
        data['subjects'],
        data['deadlines'],
        data['hours_needed'],
        data.get('subject_names', None)
    )
    greedy = greedy_planner(tasks, data['avail_hours'])
    astar  = astar_planner(tasks,  data['avail_hours'])

    # Display
    show_results(data, ml_result, fuzzy_result,
                 fired_rules, bwd_result, greedy, astar)

# -------------------------------------------------------
# TWO DEMO SCENARIOS
# -------------------------------------------------------
def run_demo(scenario):
    """
    Pre-defined scenarios for demonstration.
    Scenario 1 — Alex: struggling student, CRITICAL risk
    Scenario 2 — Jordan: well-prepared student, LOW risk
    """
    scenarios = {
        1: {
            'name':             'Alex',
            'gender':           'female',
            'year':             '3',
            'subjects':         7,
            'subject_names':    ['Mathematics', 'Computer Science',
                                 'Physics', 'History', 'Economics',
                                 'Chemistry', 'Literature'],
            'deadlines':        4,
            'days_to_deadline': 2,
            'hours_needed':     30,
            'avail_hours':      2.0,
            'confidence':       20,
            'stress':           9,
            'sleep':            5.0,
            'total_sessions':   10,
            'missed_sessions':  4
        },
        2: {
            'name':             'Jordan',
            'gender':           'male',
            'year':             '2',
            'subjects':         3,
            'subject_names':    ['Biology', 'Statistics', 'Psychology'],
            'deadlines':        1,
            'days_to_deadline': 6,
            'hours_needed':     10,
            'avail_hours':      6.0,
            'confidence':       85,
            'stress':           2,
            'sleep':            8.0,
            'total_sessions':   8,
            'missed_sessions':  0
        }
    }
    _run_analysis(scenarios[scenario])

# -------------------------------------------------------
# MAIN MENU
# -------------------------------------------------------
def main_menu():
    while True:
        clear()
        header("🎓 STUDENT SUCCESS COPILOT")
        print("  AI-Powered Study Planning System\n")
        print("  [1]  Enter my details manually")
        print("  [2]  Demo: Alex   — Struggling Student")
        print("  [3]  Demo: Jordan — Well-Prepared Student")
        print("  [4]  Exit\n")
        line()

        choice = input("  Choose (1-4): ").strip()

        if not choice:
            continue
        elif choice == '1':
            data = get_inputs()
            data = ask_followups(data)
            _run_analysis(data)
        elif choice == '2':
            run_demo(1)
        elif choice == '3':
            run_demo(2)
        elif choice == '4':
            print("\n  Goodbye! Good luck! 🎓\n")
            break
        else:
            input("  Invalid choice. Press Enter to try again...")