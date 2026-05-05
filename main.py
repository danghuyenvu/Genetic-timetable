import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import time
DAY_NAME = {2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri"}

# --- CONFIGURATION ---
NUM_COURSES = 15
SECTIONS_PER_COURSE = 10
NUM_PROFESSORS = 10
DAYS = [2, 3, 4, 5, 6]  # Mon, Tue, Wed, Thu, Fri
TIMESLOTS = [1, 2, 3, 4, 5, 6]
NUM_ROOMS = 30
REPAIR_TIMES = 20

# Room Sizes: 0 for Small, 1 for Large
ROOM_SIZES = [random.randint(0, 1) for r in range(1, NUM_ROOMS + 1)]
ROOM_BIG = []

CLASS_SIZE = [random.randint(0, 1) for r in range(NUM_COURSES)]

# --- GA SETTINGS ---
POP_SIZE = 200
GEN = 400
MUTATION_RATE = 0.2
TOP_ELITE = int(0.05 * POP_SIZE)

for room in range(NUM_ROOMS):
    if ROOM_SIZES[room] == 1:
        ROOM_BIG.append(room + 1)

def build_professor_map():
    """
    Returns prof_map[course_id][section_id] = professor_id

    Strategy: assign courses to professors in round-robin,
    respecting the max 3 courses per professor constraint.
    With 15 courses and 10 profs, each prof gets exactly 1 or 2 courses.
    Sections 1-5 go to prof A, sections 6-10 go to prof B for each course.
    """
    prof_courses = defaultdict(list)  # prof -> [course_ids]
    course_profs = {}                 # course_id -> [profA, profB]

    prof_id = 1
    for course in range(1, NUM_COURSES + 1):
        # Find two professors who still have room (< 3 courses)
        assigned = []
        attempts = 0
        p = prof_id
        while len(assigned) < 2:
            if len(prof_courses[p]) < 3:
                assigned.append(p)
                prof_courses[p].append(course)
            p = (p % NUM_PROFESSORS) + 1
            attempts += 1
            if attempts > NUM_PROFESSORS * 3:
                break  # fallback: shouldn't happen with these numbers
        course_profs[course] = assigned
        prof_id = p

    # Build final map: section 1-5 → profA, section 6-10 → profB
    prof_map = {}
    for course, (profA, profB) in course_profs.items():
        prof_map[course] = {}
        for sec in range(1, SECTIONS_PER_COURSE + 1):
            prof_map[course][sec] = profA if sec <= 5 else profB

    return prof_map, prof_courses

PROF_MAP, PROF_COURSES = build_professor_map()

# --- DATA CLASSES ---
class Slot:
    def __init__(self, day, timeslot):
        self.day = day
        self.timeslot = timeslot

PROFESSOR_SLOT = {r: [Slot(i, j) for i in DAYS for j in TIMESLOTS] for r in range(1, NUM_PROFESSORS + 1)}

class Section:
    def __init__(self, course_id, section_id):
        self.course_id = course_id
        self.section_id = section_id
        # FIXED ATTRIBUTES
        self.size = CLASS_SIZE[course_id - 1]  # 0: Small registration, 1: Large
        self.lecturer = PROF_MAP[course_id][section_id]

        # VARIABLE ATTRIBUTES (The Genes)
        self.room = random.randint(1, NUM_ROOMS) if self.size == 0 else random.choice(ROOM_BIG)
        self.slot1 = random.choice(PROFESSOR_SLOT[self.lecturer])
        remaining = [i for i in PROFESSOR_SLOT[self.lecturer] if abs(i.day - self.slot1.day) > 1]
        self.slot2 = random.choice(remaining)

    def copy(self):
        new_sec = object.__new__(Section)   # skip __init__ entirely
        new_sec.course_id  = self.course_id
        new_sec.section_id = self.section_id
        new_sec.size       = self.size
        new_sec.lecturer   = self.lecturer
        new_sec.room       = self.room
        new_sec.slot1      = Slot(self.slot1.day, self.slot1.timeslot)
        new_sec.slot2      = Slot(self.slot2.day, self.slot2.timeslot)
        return new_sec

    def mutate(self):
        choice = random.randint(0, 1)
        if choice == 0:
            # mutate the room choice
            self.room = random.randint(1, NUM_ROOMS) if self.size == 0 else random.choice(ROOM_BIG)
        else:
            # mutate the slot choice
            self.slot1 = random.choice(PROFESSOR_SLOT[self.lecturer])
            remaining = [i for i in PROFESSOR_SLOT[self.lecturer] if abs(i.day - self.slot1.day) > 1]
            self.slot2 = random.choice(remaining)

def calculate_fitness(chromosome):
    conflicts = 0
    # Track usage: (Day, Slot) -> Set of Rooms/Profs used
    room_usage = set()
    prof_usage = set()

    for sec in chromosome:
        for s in (sec.slot1, sec.slot2):
            r_key = (s.day, s.timeslot, sec.room)
            p_key = (s.day, s.timeslot, sec.lecturer)

            if r_key in room_usage: conflicts += 1
            else: room_usage.add(r_key)

            if p_key in prof_usage: conflicts += 1
            else: prof_usage.add(p_key)

    return 0 - conflicts

def UniformCross(daddy, mommy):
    bits = random.getrandbits(150)
    return [
        (daddy[i] if (bits >> i) & 1 else mommy[i]).copy()
        for i in range(150)
    ]

def repair(chromosome):
    """Reassign slots for sections whose professor has a time conflict."""
    prof_usage = {}  # (day, slot, prof) -> section index

    for i, sec in enumerate(chromosome):
        for slot in [sec.slot1, sec.slot2]:
            key = (slot.day, slot.timeslot, sec.lecturer)
            if key in prof_usage:
                # Conflict found — reassign this section's slots
                for _ in range(REPAIR_TIMES):  # try up to 20 random reassignments
                    new_slot1 = random.choice(PROFESSOR_SLOT[sec.lecturer])
                    remaining = [i for i in PROFESSOR_SLOT[sec.lecturer] if abs(i.day - new_slot1.day) > 1]
                    if not remaining:
                        continue
                    new_slot2 = random.choice(remaining)
                    k1 = (new_slot1.day, new_slot1.timeslot, sec.lecturer)
                    k2 = (new_slot2.day, new_slot2.timeslot, sec.lecturer)
                    if k1 not in prof_usage and k2 not in prof_usage:
                        sec.slot1 = new_slot1
                        sec.slot2 = new_slot2
                        prof_usage[k1] = i
                        prof_usage[k2] = i
                        break
            else:
                prof_usage[key] = i
    return chromosome

# Choose top candidate from a tournament
def Tournament(population, fitness, tour_size=5):
    contestants = random.sample(range(len(population)), tour_size)
    winner = max(contestants, key=lambda i: fitness[i])
    return population[winner]

def diagnose(chromosome):
    room_c = prof_c = size_c = day_c = 0
    room_usage, prof_usage = {}, {}
    for sec in chromosome:
        if sec.size == 1 and ROOM_SIZES[sec.room - 1] == 0:
            size_c += 1
        if abs(sec.slot1.day - sec.slot2.day) < 2:
            day_c += 1
        for d, sl in [(sec.slot1.day, sec.slot1.timeslot), (sec.slot2.day, sec.slot2.timeslot)]:
            r_key, p_key = (d, sl, sec.room), (d, sl, sec.lecturer)
            if r_key in room_usage: room_c += 1
            else: room_usage[r_key] = True
            if p_key in prof_usage: prof_c += 1
            else: prof_usage[p_key] = True
    print(f"  Room overlaps:    {room_c}")
    print(f"  Prof overlaps:    {prof_c}")
    print(f"  Score:            {-(room_c + prof_c)}")

def print_professor_tables(chromosome):
    # Group sections by lecturer
    prof_sections = defaultdict(list)
    for s in chromosome:
        prof_sections[s.lecturer].append(s)

    for prof_id in sorted(prof_sections.keys()):
        sections = prof_sections[prof_id]
        print(f"\n{'='*72}")
        print(f"  PROFESSOR {prof_id}")
        print(f"{'='*72}")
        print(f"{'Course-Sec':<12} {'Session':<10} {'Day':<6} {'Slot':<6} {'Room':<6} {'RmSz':<6} {'ClSz'}")
        print(f"{'-'*72}")

        # Sort by course then section for readability
        for s in sorted(sections, key=lambda x: (x.course_id, x.section_id)):
            course_sec = f"C{s.course_id}-S{s.section_id}"
            room_size  = ROOM_SIZES[s.room - 1]

            print(f"{course_sec:<12} {'Session 1':<10} {DAY_NAME[s.slot1.day]:<6} {s.slot1.timeslot:<6} R{s.room:<5} {room_size:<6} {s.size}")
            print(f"{'.'*12} {'Session 2':<10} {DAY_NAME[s.slot2.day]:<6} {s.slot2.timeslot:<6} R{s.room:<5} {room_size:<6} {s.size}")

        print(f"{'-'*72}")
        print(f"  Total sections: {len(sections)}")

# Initialize Population
population = [[Section(c, s) for c in range(1, 16) for s in range(1, 11)] for _ in range(POP_SIZE)]
history = []
history_average = []
best_fit = 0.0
total_generations = 0
GOAT_table = None
GOAT_score = -float("inf")

if __name__ == "__main__":
    start = time.perf_counter()
    while total_generations < GEN:
        # Sort by fitness
        fitnesses = [calculate_fitness(chrom) for chrom in population]
        ranked = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)

        best_fit = fitnesses[ranked[0]]
        history_average.append((sum(fitnesses) / len(fitnesses)))
        history.append(best_fit)
        if best_fit > GOAT_score:
            GOAT_score = best_fit
            GOAT_table = population[ranked[0]]

        # Elitism: Keep the best schedule
        next_gen = [population[i] for i in ranked[:TOP_ELITE]]

        while len(next_gen) < POP_SIZE:
            # Selection (Tournament)
            p1, p2 = Tournament(population, fitnesses), Tournament(population, fitnesses)

            # Uniform crossover:
            child = UniformCross(p1, p2)

            # Mutation: Give each section a chance to mutate
            for sec in child:
                if random.random() < MUTATION_RATE:
                    sec.mutate()
            
            repair(child)

            next_gen.append(child)

        population = next_gen
        total_generations += 1
        if total_generations % 50 == 0:
            print(f"Current generation: {total_generations}. Current best_fit: {GOAT_score}")
            cur = time.perf_counter()
            print(f"It's been {cur - start:.4f} seconds from the start of the execution")
    
    end = time.perf_counter()
    print(f"Total {end - start:.4f} seconds for {GEN} generations")

    # --- 4. RESULTS ---
    # Create a figure with 1 row and 2 columns of subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First plot: history
    axes[0].plot(history)
    axes[0].set_title("Fitness Evolution")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness Score")

    # Second plot: history_average
    axes[1].plot(history_average, color="orange")
    axes[1].set_title("Average Fitness Evolution")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Average Fitness Score")

    plt.tight_layout()
    plt.show()

    # Display snippet of best schedule
    diagnose(GOAT_table)
    print(f"Best Fitness score: {GOAT_score}")
    # print("Best Schedule (Course-Section | Prof | Day1-Slot1 | Day2-Slot2 | Room | Room Size | class size)")
    # for s in GOAT_table:
    #     print(f"C{s.course_id}-S{s.section_id} | P{s.lecturer} | {s.slot1.day}-{s.slot1.timeslot} | {s.slot2.day}-{s.slot2.timeslot} | R{s.room} | {ROOM_SIZES[s.room - 1]} | {s.size}")
    print_professor_tables(GOAT_table)