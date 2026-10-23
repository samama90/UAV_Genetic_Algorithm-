import matplotlib.pyplot as plt
import numpy as np
from environment import DisasterEnvironment
from genetic_algorithm import GeneticAlgorithm
from chromosome import Chromosome

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def run_ga_experiment(env, pop_size, mut_rate, total_evals, selection_method="roulette", t_size=3):
    """
    Runs GA with strict evaluation budget and selectable selection method.
    """
    ga = GeneticAlgorithm(environment=env, population_size=pop_size, max_flight_range=150.0)
    ga.initialize_population()
    
    # Repair initial population
    for chrom in ga.population:
        ga.repair_chromosome(chrom)

    eval_axis, history = [], []
    best_so_far = -float('inf')
    evaluations = 0
    
    # Exact Budget Tracking Loop
    while evaluations + pop_size <= total_evals:
        ga.evaluate_population()
        evaluations += pop_size
        
        ga.population.sort(key=lambda x: x.fitness, reverse=True)
        if ga.population[0].fitness > best_so_far:
            best_so_far = ga.population[0].fitness
            
        eval_axis.append(evaluations)
        history.append(best_so_far)
        
        if evaluations == total_evals:
            break
            
        # Elitism
        new_pop = ga.population[:2]
        while len(new_pop) < ga.population_size:
            if selection_method == "tournament":
                p1 = ga.tournament_selection(t_size)
                p2 = ga.tournament_selection(t_size)
            else:
                p1 = ga.roulette_wheel_selection()
                p2 = ga.roulette_wheel_selection()
                
            child = ga.crossover(p1, p2)
            child = ga.mutate(child, mutation_rate=mut_rate)
            new_pop.append(child)
        ga.population = new_pop
        
    return eval_axis, history

def run_fair_random_search(env, total_evals):
    """
    Runs Random Search with exact same constraints and repair as GA.
    """
    ga = GeneticAlgorithm(environment=env, population_size=1, max_flight_range=150.0)
    eval_axis, history = [], []
    best = -float('inf')
    
    for i in range(total_evals):
        chrom = Chromosome(env, num_drones=3, waypoints_per_drone=5)
        
        # Repair random chromosome (Fairness)
        ga.repair_chromosome(chrom) 
        
        fit, _ = ga._calculate_fitness(chrom) 
        
        if fit > best: 
            best = fit
            
        eval_axis.append(i + 1)
        history.append(best)
        
    return eval_axis, history

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        env = DisasterEnvironment()
        
        # --------------------------------------------------
        # PART 17: RANDOM SEARCH BASELINE
        # --------------------------------------------------
        TOTAL_EVALS_17 = 400
        print("--- PART 17: GA vs Random Search ---")
        
        print(f"Running GA ({TOTAL_EVALS_17} evals)...")
        ga_evals, ga_history = run_ga_experiment(env, pop_size=20, mut_rate=0.15, total_evals=TOTAL_EVALS_17)

        print(f"Running Random Search ({TOTAL_EVALS_17} evals)...")
        rand_evals, random_history = run_fair_random_search(env, total_evals=TOTAL_EVALS_17)

        # Plot 1
        plt.figure(figsize=(10, 5))
        plt.plot(ga_evals, ga_history, label="Genetic Algorithm", color="blue", linewidth=2)
        plt.plot(rand_evals, random_history, label="Random Search", color="red", linestyle="--")
        plt.title("Part 17: GA vs Random Search (Exact Budget & Constraints)", fontweight='bold')
        plt.xlabel("Number of Evaluations")
        plt.ylabel("Best-so-far Fitness")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()

        # --------------------------------------------------
        # PART 18: HYPERPARAMETER EXPERIMENTS
        # --------------------------------------------------
        TOTAL_EVALS_18 = 600 
        print("\n--- PART 18: Hyperparameter Sensitivity ---")
        
        print("Running Baseline (Pop=20, Mut=0.15, Roulette)...")
        base_x, baseline = run_ga_experiment(env, pop_size=20, mut_rate=0.15, total_evals=TOTAL_EVALS_18)
        
        print("Running Pop Size = 10...")
        p10_x, pop_10 = run_ga_experiment(env, pop_size=10, mut_rate=0.15, total_evals=TOTAL_EVALS_18)
        
        print("Running Pop Size = 30...")
        p30_x, pop_30 = run_ga_experiment(env, pop_size=30, mut_rate=0.15, total_evals=TOTAL_EVALS_18)
        
        print("Running Mutation Rate = 0.05...")
        mlow_x, mut_low = run_ga_experiment(env, pop_size=20, mut_rate=0.05, total_evals=TOTAL_EVALS_18)
        
        print("Running Mutation Rate = 0.40...")
        mhigh_x, mut_high = run_ga_experiment(env, pop_size=20, mut_rate=0.40, total_evals=TOTAL_EVALS_18)

        # Selection Pressure Experiments (As requested by the review)
        print("Running Selection Pressure: Tournament (k=2)...")
        t2_x, tour_k2 = run_ga_experiment(env, pop_size=20, mut_rate=0.15, total_evals=TOTAL_EVALS_18, selection_method="tournament", t_size=2)
        
        print("Running Selection Pressure: Tournament (k=5)...")
        t5_x, tour_k5 = run_ga_experiment(env, pop_size=20, mut_rate=0.15, total_evals=TOTAL_EVALS_18, selection_method="tournament", t_size=5)

        # Plot 2: Hyperparameters (3 Subplots)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Population Size
        ax1.plot(base_x, baseline, label="Baseline (Pop=20)", color="blue")
        ax1.plot(p10_x, pop_10, label="Pop Size = 10", color="green", linestyle="--")
        ax1.plot(p30_x, pop_30, label="Pop Size = 30", color="purple", linestyle=":")
        ax1.set_title("Impact of Population Size", fontweight='bold')
        ax1.set_xlabel("Evaluations")
        ax1.set_ylabel("Best-so-far Fitness")
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # Mutation Rate
        ax2.plot(base_x, baseline, label="Baseline (Mut=0.15)", color="blue")
        ax2.plot(mlow_x, mut_low, label="Mutation Rate = 0.05", color="orange", linestyle="--")
        ax2.plot(mhigh_x, mut_high, label="Mutation Rate = 0.40", color="brown", linestyle=":")
        ax2.set_title("Impact of Mutation Rate", fontweight='bold')
        ax2.set_xlabel("Evaluations")
        ax2.set_ylabel("Best-so-far Fitness")
        ax2.legend()
        ax2.grid(True, alpha=0.5)

        # Selection Pressure
        ax3.plot(base_x, baseline, label="Roulette Wheel", color="blue")
        ax3.plot(t2_x, tour_k2, label="Tournament (k=2, Low)", color="teal", linestyle="--")
        ax3.plot(t5_x, tour_k5, label="Tournament (k=5, High)", color="darkred", linestyle="-.")
        ax3.set_title("Impact of Selection Pressure", fontweight='bold')
        ax3.set_xlabel("Evaluations")
        ax3.set_ylabel("Best-so-far Fitness")
        ax3.legend()
        ax3.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

        # Success message ONLY if everything runs without error
        print("\nAll Experiments Completed Successfully!")

    except Exception as e:
        print(f"\nError occurred during experiments: {e}")
        print("Experiments did not complete successfully.")