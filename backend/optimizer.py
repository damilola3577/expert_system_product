import numpy as np

MIN_SP = 18
MAX_SP = 26
HOURS = 24

def energy_for_setpoint(setpoint, outdoor_temp=10.0):
    base = 0.2
    alpha = 0.08
    return base + alpha * max(0, setpoint - outdoor_temp)

def comfort_penalty(schedule, desired=21):
    diffs = np.abs(schedule - desired)
    weights = 1 / (1 + np.exp(- (diffs - 1.0)))
    return float(np.sum(weights * diffs))

def apply_rules(schedule):
    schedule = np.clip(schedule, MIN_SP, MAX_SP)
    return schedule

def random_population(pop_size):
    return np.random.randint(MIN_SP, MAX_SP + 1, size=(pop_size, HOURS))

def fitness(schedule, tariff, desired=21, outdoor_temp=10.0, comfort_weight=1.0):
    hourly_energy = np.array([energy_for_setpoint(int(sp), outdoor_temp) for sp in schedule])
    cost = np.sum(hourly_energy * tariff)
    penalty = comfort_penalty(schedule, desired)
    return -(cost + comfort_weight * penalty)

def tournament_select(pop, scores, k=3):
    idx = np.random.randint(0, len(pop), k)
    best = idx[np.argmax(scores[idx])]
    return pop[best].copy()

def crossover(a, b):
    point = np.random.randint(1, HOURS-1)
    child = np.concatenate([a[:point], b[point:]])
    return child

def mutate(child, mutation_rate=0.02):
    for i in range(HOURS):
        if np.random.rand() < mutation_rate:
            child[i] = np.random.randint(MIN_SP, MAX_SP+1)
    return child

def run_ga(tariff, pop_size=50, gens=120, desired=21, comfort_weight=1.0, outdoor_temp=10.0):
    pop = random_population(pop_size)
    for g in range(gens):
        scores = np.array([fitness(ind, tariff, desired, outdoor_temp, comfort_weight) for ind in pop])
        new_pop = []
        elite_count = max(1, pop_size // 20)
        elite_idx = np.argsort(scores)[-elite_count:]
        new_pop.extend(pop[elite_idx])
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, scores)
            p2 = tournament_select(pop, scores)
            child = crossover(p1, p2)
            child = mutate(child)
            child = apply_rules(child)
            new_pop.append(child)
        pop = np.array(new_pop)
    scores = np.array([fitness(ind, tariff, desired, outdoor_temp, comfort_weight) for ind in pop])
    best = pop[np.argmax(scores)]
    hourly_energy = np.array([energy_for_setpoint(int(sp), outdoor_temp) for sp in best])
    total_cost = float(np.sum(hourly_energy * tariff))
    comfort = float(comfort_penalty(best, desired))
    return {
        "schedule": best.tolist(),
        "total_cost": total_cost,
        "comfort_penalty": comfort
    }
