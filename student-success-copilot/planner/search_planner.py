# planner/search_planner.py

import heapq
import time

SUBJECT_NAMES = [
    "Mathematics", "Computer Science", "Physics", "History",
    "Economics", "Chemistry", "Literature", "Psychology",
    "Statistics", "Biology", "Engineering", "Philosophy"
]

def build_tasks(num_subjects, num_deadlines, hours_needed,
                subject_names=None):
    """
    Builds task list using real subject names if provided.
    Caps hours per subject so all subjects get allocated time.
    """
    tasks = []
    base_hours = hours_needed / num_subjects

    for i in range(num_subjects):
        has_deadline = i < num_deadlines

        if subject_names and i < len(subject_names):
            name = subject_names[i]
        else:
            name = SUBJECT_NAMES[i % len(SUBJECT_NAMES)]

        raw_hours = base_hours * (1.2 if has_deadline else 0.8)
        capped    = round(min(raw_hours, hours_needed * 0.4), 1)

        task = {
            'id':           i + 1,
            'name':         name,
            'hours':        max(1.0, capped),
            'deadline_day': i + 1 if has_deadline else
                            7 + (i - num_deadlines + 1),
            'difficulty':   (i % 3) + 1
        }
        tasks.append(task)
    return tasks


def greedy_planner(tasks, avail_hours_per_day):
    """
    SEARCH METHOD 1: Greedy Best-First
    Sorts tasks by urgency = difficulty / deadline_day
    Fast but only locally optimal.
    """
    start = time.perf_counter()

    sorted_tasks = sorted(
        tasks,
        key=lambda t: t['difficulty'] / (t['deadline_day'] + 0.1),
        reverse=True
    )

    schedule, stats = _allocate(sorted_tasks, avail_hours_per_day)
    elapsed = round((time.perf_counter() - start) * 1000, 3)

    return {
        'method':     'Greedy Best-First',
        'schedule':   schedule,
        'nodes':      stats['nodes'],
        'backtracks': stats['backtracks'],
        'time_ms':    elapsed
    }


def astar_planner(tasks, avail_hours_per_day):
    """
    SEARCH METHOD 2: A* Heuristic
    f(n) = g(n) + h(n)
    g(n) = 0 (no prior cost)
    h(n) = hours / deadline_day (urgency estimate)
    More globally optimal than greedy.
    """
    start = time.perf_counter()

    heap = []
    for i, task in enumerate(tasks):
        h = task['hours'] / (task['deadline_day'] + 0.1)
        heapq.heappush(heap, (-h, i, task))

    sorted_tasks  = []
    nodes_explored = 0
    while heap:
        _, _, task = heapq.heappop(heap)
        sorted_tasks.append(task)
        nodes_explored += 1

    schedule, stats = _allocate(sorted_tasks, avail_hours_per_day)
    stats['nodes']   = nodes_explored
    elapsed = round((time.perf_counter() - start) * 1000, 3)

    return {
        'method':     'A* Heuristic',
        'schedule':   schedule,
        'nodes':      stats['nodes'],
        'backtracks': stats['backtracks'],
        'time_ms':    elapsed
    }


def _allocate(sorted_tasks, avail_hours_per_day):
    """
    Allocates tasks across 7 days.
    Called by both planners — same allocation, different order.
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday']

    schedule  = {day: [] for day in days}
    remaining = {t['name']: t['hours'] for t in sorted_tasks}
    nodes     = 0
    backtracks = 0

    for day in days:
        day_budget = avail_hours_per_day
        for task in sorted_tasks:
            if day_budget <= 0:
                break
            name = task['name']
            if remaining[name] <= 0:
                continue
            alloc = min(day_budget, remaining[name], 2.5)
            if alloc > 0:
                schedule[day].append({
                    'task':  name,
                    'hours': round(alloc, 1)
                })
                remaining[name] -= alloc
                day_budget      -= alloc
                nodes           += 1
                if remaining[name] > 0 and day_budget <= 0:
                    backtracks += 1

    return schedule, {'nodes': nodes, 'backtracks': backtracks}


if __name__ == "__main__":
    tasks  = build_tasks(
        num_subjects  = 4,
        num_deadlines = 2,
        hours_needed  = 20,
        subject_names = ['Maths', 'Economics', 'Chemistry', 'History']
    )

    greedy = greedy_planner(tasks, avail_hours_per_day=2.5)
    astar  = astar_planner(tasks,  avail_hours_per_day=2.5)

    for result in [greedy, astar]:
        print(f"\n=== {result['method']} ===")
        print(f"Nodes: {result['nodes']} | "
              f"Backtracks: {result['backtracks']} | "
              f"Time: {result['time_ms']}ms")
        for day, slots in result['schedule'].items():
            if slots:
                items = ', '.join(
                    f"{s['task']} ({s['hours']}h)" for s in slots
                )
                print(f"  {day}: {items}")