import heapq
import math
import numpy as np
from environment import DisasterEnvironment
from chromosome import Chromosome


class GeneticAlgorithm:
    def __init__(
        self,
        environment,
        population_size=20,
        num_drones=3,
        waypoints_per_drone=5,
        max_flight_range=100.0
    ):
        self.env = environment
        self.population_size = population_size
        self.num_drones = num_drones
        self.waypoints_per_drone = waypoints_per_drone
        self.max_flight_range = max_flight_range

        # ISSUE 4 FIX: Check if enough drones exist in environment
        if self.num_drones > len(self.env.drone_starts):
            raise ValueError(f"Environment has only {len(self.env.drone_starts)} drones, but {self.num_drones} requested.")

        self.population = []
        self.best_chromosome = None
        self.best_fitness = -float("inf")

        # Fitness weights
        self.w_coverage = 10.0
        self.w_detection = 10.0
        self.w_energy = 2.0
        self.w_time = 1.0
        self.w_hazard = 5.0

        # Constraint penalties
        self.no_fly_penalty = 1000.0
        self.range_penalty = 500.0

    # Part 3: Initial Population
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = Chromosome(
                environment=self.env,
                num_drones=self.num_drones,
                waypoints_per_drone=self.waypoints_per_drone
            )
            self.population.append(chromosome)
        print(f"Initial population of {self.population_size} chromosomes generated.")

    # Part 4: Constraint Handling
    def is_valid_position(self, x, y):
        if not (0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size):
            return False
        return self.env.no_fly_zones[y, x] == 0

    def check_segment(self, start, end):
        x1, y1 = start
        x2, y2 = end
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        steps = max(1, math.ceil(distance * 2))
        for step in range(steps + 1):
            ratio = step / steps
            x = round(x1 + (x2 - x1) * ratio)
            y = round(y1 + (y2 - y1) * ratio)
            if not self.is_valid_position(x, y):
                return False
        return True

    # Part 5: Fitness Function
    def _calculate_fitness(self, chromosome):
        total_energy, total_time, total_hazard_exposure, total_distance = 0.0, 0.0, 0.0, 0.0
        no_fly_violations, range_violations = 0, 0
        drone_distances = []
        covered_cells, detection_scores = set(), {}
        total_priority_cells = int(self.env.priority_zones.sum())

        for drone_idx in range(self.num_drones):
            mission = chromosome.get_drone_mission(drone_idx)
            drone_name = list(self.env.drone_starts.keys())[drone_idx]
            current_pos = self.env.drone_starts[drone_name]
            drone_distance = 0.0

            for wp in mission:
                next_pos = (wp["x"], wp["y"])
                path = self.get_safe_path(current_pos, next_pos)
                
                # ISSUE 3 FIX: Update current_pos even if path is None
                if path is None:
                    no_fly_violations += 1
                    current_pos = next_pos 
                    continue 

                speed = max(wp["speed"], 0.01)
                sensor_power = wp["sensor_power"]

                for i in range(len(path) - 1):
                    p2 = path[i+1]
                    step_distance = 1.0 
                    drone_distance += step_distance
                    total_distance += step_distance

                    energy = step_distance * (1.0 + speed * 0.2)
                    energy += step_distance * sensor_power * 0.1
                    total_energy += energy
                    total_time += step_distance / speed

                    if self.env.hazard_zones[p2[1], p2[0]] == 1:
                        total_hazard_exposure += 1.0

                if wp["sensor_active"] == 1:
                    radius = 1.0 + 2.0 * sensor_power
                    min_x = max(0, math.floor(next_pos[0] - radius))
                    max_x = min(self.env.grid_size - 1, math.ceil(next_pos[0] + radius))
                    min_y = max(0, math.floor(next_pos[1] - radius))
                    max_y = min(self.env.grid_size - 1, math.ceil(next_pos[1] + radius))

                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            if not ((x - next_pos[0])**2 + (y - next_pos[1])**2 <= radius**2):
                                continue
                            if self.env.priority_zones[y, x] == 1:
                                cell = (x, y)
                                covered_cells.add(cell)
                                detection_scores[cell] = max(detection_scores.get(cell, 0.0), sensor_power)

                current_pos = next_pos

            drone_distances.append(drone_distance)
            if drone_distance > self.max_flight_range:
                range_violations += 1

        coverage = len(covered_cells)
        if total_priority_cells > 0:
            coverage_rate = (coverage / total_priority_cells) * 100.0
            detection_rate = (sum(detection_scores.values()) / total_priority_cells) * 100.0
        else:
            coverage_rate = 0.0
            detection_rate = 0.0

        fitness = (
            coverage_rate * self.w_coverage + detection_rate * self.w_detection
            - total_energy * self.w_energy - total_time * self.w_time - total_hazard_exposure * self.w_hazard
        )
        fitness -= no_fly_violations * self.no_fly_penalty
        fitness -= range_violations * self.range_penalty

        metrics = {
            "coverage": coverage, "coverage_rate": coverage_rate, "detection_rate": detection_rate,
            "energy": total_energy, "time": total_time, "hazard": total_hazard_exposure,
            "total_distance": total_distance, "drone_distances": drone_distances,
            "no_fly_violations": no_fly_violations, "range_violations": range_violations
        }
        return fitness, metrics

    # Part 6: Roulette Wheel Selection
    def roulette_wheel_selection(self):
        if not self.population:
            raise ValueError("Population is empty. Initialize it first.")

        fitness_scores = [chromosome.fitness for chromosome in self.population]
        min_fitness = min(fitness_scores)

        if min_fitness <= 0:
            offset = abs(min_fitness) + 1.0
            adjusted_scores = [score + offset for score in fitness_scores]
        else:
            adjusted_scores = fitness_scores

        total_fitness = sum(adjusted_scores)
        if total_fitness <= 0:
            return self.population[np.random.randint(len(self.population))]

        pick = np.random.uniform(0, total_fitness)
        current_sum = 0.0

        for i, chromosome in enumerate(self.population):
            current_sum += adjusted_scores[i]
            if current_sum > pick:
                return chromosome
        return self.population[-1]

    # FIXED INDENTATION HERE
    def tournament_selection(self, tournament_size=3):
        if not self.population:
            raise ValueError("Population is empty. Initialize it first.")

        k = min(tournament_size, len(self.population))
        indices = np.random.choice(
            len(self.population), size=k, replace=False
        )
        candidates = [self.population[i] for i in indices]

        return max(candidates, key=lambda c: c.fitness)

    def select_parents(self, num_parents):
        parents = []
        for _ in range(num_parents):
            parents.append(self.roulette_wheel_selection())
        return parents

    def evaluate_population(self):
        for chromosome in self.population:
            fitness, metrics = self._calculate_fitness(chromosome)
            chromosome.fitness = fitness
            chromosome.metrics = metrics

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_chromosome = chromosome
        print("Population fitness evaluation completed.")

    # A*: Safe Navigation Between Waypoints
    def get_safe_path(self, start, goal):
        if not self.is_valid_position(*start):
            return None
        if not self.is_valid_position(*goal):
            return None

        open_list = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_list:
            _, current = heapq.heappop(open_list)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            x, y = current
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

            for neighbor in neighbors:
                if not self.is_valid_position(*neighbor):
                    continue

                new_cost = g_score[current] + 1
                if new_cost < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_cost
                    came_from[neighbor] = current
                    heuristic = abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                    heapq.heappush(open_list, (new_cost + heuristic, neighbor))
        return None

    # Part 7 & 8: Custom Crossover + Constraints
    def crossover(self, parent1, parent2):
        child = Chromosome(
            environment=self.env,
            num_drones=self.num_drones,
            waypoints_per_drone=self.waypoints_per_drone
        )

        for drone_idx in range(self.num_drones):
            for wp_idx in range(self.waypoints_per_drone):
                start_idx = (drone_idx * self.waypoints_per_drone + wp_idx) * child.genes_per_wp
                end_idx = start_idx + child.genes_per_wp

                if wp_idx == 0:
                    child.genes[start_idx:end_idx] = parent1.genes[start_idx:end_idx]
                elif np.random.rand() < 0.5:
                    child.genes[start_idx:end_idx] = parent1.genes[start_idx:end_idx]
                else:
                    child.genes[start_idx:end_idx] = parent2.genes[start_idx:end_idx]

        self.repair_chromosome(child)
        child.fitness = 0.0
        child.metrics = {}
        return child

    # Part 9 & 10: Mutation + Constraints
    def mutate(self, chromosome, mutation_rate=0.1):
        for drone_idx in range(self.num_drones):
            for wp_idx in range(self.waypoints_per_drone):
                if wp_idx == 0:
                    continue

                start_idx = (drone_idx * self.waypoints_per_drone + wp_idx) * chromosome.genes_per_wp

                if np.random.rand() < mutation_rate:
                    gene = np.random.randint(0, chromosome.genes_per_wp)

                    if gene in [0, 1]:
                        x, y = chromosome._get_random_valid_coordinate()
                        chromosome.genes[start_idx] = x
                        chromosome.genes[start_idx + 1] = y
                    elif gene == 2:
                        chromosome.genes[start_idx + 2] = np.random.uniform(1.0, 5.0)
                    elif gene == 3:
                        chromosome.genes[start_idx + 3] = np.random.uniform(0.1, 1.0)
                    elif gene == 4:
                        chromosome.genes[start_idx + 4] = 1 - round(chromosome.genes[start_idx + 4])

        self.repair_chromosome(chromosome)
        chromosome.fitness = 0.0
        chromosome.metrics = {}
        return chromosome

    # Helper: Calculate Actual Safe Route Distance
    def route_distance(self, route):
        distance = 0.0
        for i in range(1, len(route)):
            path = self.get_safe_path(route[i - 1], route[i])
            if path is None:
                return float("inf")
            distance += len(path) - 1
        return distance

    # Helper: Repair Waypoints, Routes and Range
    def repair_chromosome(self, chromosome):
        drone_names = list(self.env.drone_starts.keys())

        for drone_idx in range(self.num_drones):
            drone_name = drone_names[drone_idx]
            start = self.env.drone_starts[drone_name]
            points = []

            # Step 1: Repair waypoint coordinates
            for wp_idx in range(self.waypoints_per_drone):
                idx = (drone_idx * self.waypoints_per_drone + wp_idx) * chromosome.genes_per_wp

                if wp_idx == 0:
                    x, y = start
                    chromosome.genes[idx] = x
                    chromosome.genes[idx + 1] = y
                else:
                    x = int(round(chromosome.genes[idx]))
                    y = int(round(chromosome.genes[idx + 1]))
                    if not self.is_valid_position(x, y):
                        x, y = chromosome._get_random_valid_coordinate()
                    chromosome.genes[idx] = x
                    chromosome.genes[idx + 1] = y

                points.append((x, y))

            # Step 2: Repair unreachable waypoint connections
            for wp_idx in range(1, self.waypoints_per_drone):
                previous = points[wp_idx - 1]
                current = points[wp_idx]
                path = self.get_safe_path(previous, current)

                if path is None:
                    idx = (drone_idx * self.waypoints_per_drone + wp_idx) * chromosome.genes_per_wp
                    repaired = False

                    for _ in range(100):
                        new_x, new_y = chromosome._get_random_valid_coordinate()
                        new_point = (new_x, new_y)

                        if self.get_safe_path(previous, new_point) is not None:
                            chromosome.genes[idx] = new_x
                            chromosome.genes[idx + 1] = new_y
                            points[wp_idx] = new_point
                            repaired = True
                            break

                    if not repaired:
                        continue

            # Step 3: Try to bring route within maximum flight range
            for _ in range(100):
                distance = self.route_distance(points)
                if distance <= self.max_flight_range:
                    break
                if self.waypoints_per_drone <= 1:
                    break

                wp_idx = np.random.randint(1, self.waypoints_per_drone)
                previous = points[wp_idx - 1]
                current = points[wp_idx]

                new_x = int(round((current[0] + previous[0]) / 2))
                new_y = int(round((current[1] + previous[1]) / 2))
                new_point = (new_x, new_y)

                if (new_point != current and self.is_valid_position(*new_point) and self.get_safe_path(previous, new_point) is not None):
                    next_is_safe = (wp_idx == self.waypoints_per_drone - 1 or self.get_safe_path(new_point, points[wp_idx + 1]) is not None)

                    if next_is_safe:
                        idx = (drone_idx * self.waypoints_per_drone + wp_idx) * chromosome.genes_per_wp
                        chromosome.genes[idx] = new_x
                        chromosome.genes[idx + 1] = new_y
                        points[wp_idx] = new_point
        return chromosome

    # EXTRA POINT FIX: Complete GA Loop
    def run_ga(self, generations=10, mutation_rate=0.15):
        self.initialize_population()
        print(f"\n--- Running GA Evolution for {generations} Generations ---")
        
        for gen in range(generations):
            self.evaluate_population()
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            if gen % 2 == 0 or gen == generations - 1:
                print(f"Gen {gen:2d} | Best Fitness: {self.population[0].fitness:8.2f} | Violations: {self.population[0].metrics['no_fly_violations']}")

            # Elitism
            new_pop = self.population[:2]
            while len(new_pop) < self.population_size:
                p1 = self.roulette_wheel_selection()
                p2 = self.roulette_wheel_selection()
                child = self.crossover(p1, p2)
                child = self.mutate(child, mutation_rate)
                new_pop.append(child)
            self.population = new_pop
            
        return self.best_chromosome


# ==========================================
# MAIN EXECUTION BLOCK (Fixes applied)
# ==========================================
if __name__ == "__main__":
    env = DisasterEnvironment()
    ga = GeneticAlgorithm(environment=env, population_size=10, max_flight_range=100.0)

    print("\n--- 1. Part 3: Initial Population ---")
    ga.initialize_population()

    print("\n--- 2. Part 4 & 5: Constraint Checking and Fitness ---")
    ga.evaluate_population()
    
    sample_chrom = ga.population[0]
    print(f"Sample Chromosome Fitness: {sample_chrom.fitness:.2f}")
    print(f"No-Fly Violations: {sample_chrom.metrics['no_fly_violations']}")

    print("\n--- 3. Testing A* Safe Path (get_safe_path) ---")
    # ISSUE 1 FIX: Use actual drone start and generated waypoint
    drone_name = list(env.drone_starts.keys())[0]
    start_test = env.drone_starts[drone_name]
    
    goal_mission = sample_chrom.get_drone_mission(0)[1] # Waypoint 2 of Drone 1
    goal_test = (goal_mission["x"], goal_mission["y"])
    
    print(f"Searching path from {start_test} to {goal_test}...")
    path = ga.get_safe_path(start_test, goal_test)
    if path:
        print(f"A* Path found in {len(path)} steps.")
        print(f"Path preview (first 5 steps): {path[:5]}...")
    else:
        print("A* could not find a path (Blocked).")

    print("\n--- 4. Part 6: Roulette Wheel Selection ---")
    selected_parents = ga.select_parents(num_parents=2)
    parent1, parent2 = selected_parents[0], selected_parents[1]
    print(f"Parent 1 Fitness: {parent1.fitness:.2f}")
    print(f"Parent 2 Fitness: {parent2.fitness:.2f}")

    print("\n--- 5. Part 7 & 8: Custom Crossover ---")
    child = ga.crossover(parent1, parent2)
    print("Crossover method executed. Child created.")
    print(f"Child Fitness before eval: {child.fitness} (Reset to 0.0)")

    print("\n--- 6. Part 9 & 10: Mutation ---")
    wp_before = child.get_drone_mission(0)[1]
    print(f"Drone 1 Waypoint 2 BEFORE Mutation: {wp_before}")
    
    mutated_child = ga.mutate(child, mutation_rate=1.0) 
    
    wp_after = mutated_child.get_drone_mission(0)[1]
    print(f"Drone 1 Waypoint 2 AFTER Mutation : {wp_after}")
    
    # ISSUE 2 FIX: Changed message
    print("Mutation method executed successfully.")

    print("\n--- 7. Evaluate Mutated Child ---")
    child_fitness, child_metrics = ga._calculate_fitness(mutated_child)
    print(f"New Child Fitness: {child_fitness:.2f}")
    print(f"New Child No-Fly Violations: {child_metrics['no_fly_violations']}")
    print(f"New Child Coverage Rate: {child_metrics['coverage_rate']:.2f}%")

    print("\n============================================")
    print("FINAL STEP: RUNNING COMPLETE GA EVOLUTION")
    print("============================================")
    best_mission = ga.run_ga(generations=10, mutation_rate=0.15)
    
    print("\n--- FINAL BEST MISSION RESULTS ---")
    print(f"Best Fitness Score : {best_mission.fitness:.2f}")
    print(f"Coverage Rate      : {best_mission.metrics['coverage_rate']:.2f}%")
    print(f"No-Fly Violations  : {best_mission.metrics['no_fly_violations']}")