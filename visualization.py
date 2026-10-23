
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from matplotlib.colors import ListedColormap
from environment import DisasterEnvironment
from genetic_algorithm import GeneticAlgorithm


def run_visualization():
    env = DisasterEnvironment()

    ga = GeneticAlgorithm(
        environment=env,
        population_size=20,
        max_flight_range=150.0
    )

    ga.initialize_population()

    best_history = []
    avg_history = []
    best_fitness_so_far = -float("inf")
    best_chromosome = None
    generations = 30

    # Genetic Algorithm
    for gen in range(generations):
        ga.evaluate_population()
        ga.population.sort(key=lambda c: c.fitness, reverse=True)

        current_best = ga.population[0]

        # Track the best solution found so far
        if current_best.fitness > best_fitness_so_far:
            best_fitness_so_far = current_best.fitness
            best_chromosome = copy.deepcopy(current_best)

        best_history.append(best_fitness_so_far)
        avg_history.append(
            sum(c.fitness for c in ga.population) / len(ga.population)
        )

        # Preserve the two best chromosomes
        new_population = copy.deepcopy(ga.population[:2])

        while len(new_population) < ga.population_size:
            parent1 = ga.roulette_wheel_selection()
            parent2 = ga.roulette_wheel_selection()

            child = ga.crossover(parent1, parent2)
            child = ga.mutate(child, mutation_rate=0.15)
            new_population.append(child)

        ga.population = new_population

    # Evaluate the final population
    ga.evaluate_population()
    ga.population.sort(key=lambda c: c.fitness, reverse=True)

    if ga.population[0].fitness > best_fitness_so_far:
        best_fitness_so_far = ga.population[0].fitness
        best_chromosome = copy.deepcopy(ga.population[0])

    if best_chromosome is None:
        print("No chromosome was generated.")
        return

    # Validate drone starting positions
    drone_starts = list(env.drone_starts.items())

    if len(drone_starts) != ga.num_drones:
        raise ValueError(
            f"Mismatch: {ga.num_drones} drones but "
            f"{len(drone_starts)} starting positions."
        )

    # Plot setup
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(22, 7)
    )

    colors = ["#E8E8E8", "#1A1A1A", "#FF4D4D", "#33CC33"]
    cmap = ListedColormap(colors)

    # 1. Fitness Convergence Curve
    ax1.plot(
        range(1, len(best_history) + 1),
        best_history,
        label="Best-so-far Fitness",
        color="blue",
        linewidth=2
    )

    ax1.plot(
        range(1, len(avg_history) + 1),
        avg_history,
        label="Average Fitness",
        color="orange",
        linestyle="--"
    )

    ax1.set_title("Fitness Convergence", fontweight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness Score")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # Draw environment background
    def draw_environment(ax, alpha=1.0):
        ax.imshow(
            env.grid,
            cmap=cmap,
            origin="lower",
            extent=[0, env.grid_size, 0, env.grid_size],
            interpolation="nearest",
            vmin=0,
            vmax=3,
            alpha=alpha
        )
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

    # 2. UAV Flight Paths
    draw_environment(ax2)
    drone_colors = ["blue", "orange", "purple"]

    for d_idx, (drone_name, start_pos) in enumerate(drone_starts):
        mission = best_chromosome.get_drone_mission(d_idx)
        current_pos = tuple(start_pos)
        path = []

        for wp in mission:
            next_pos = (int(wp["x"]), int(wp["y"]))
            segment = ga.get_safe_path(current_pos, next_pos)

            if not segment:
                print(
                    f"Warning: No safe path for "
                    f"{drone_name}: {current_pos} -> {next_pos}"
                )
                break

            path.extend(segment)
            current_pos = next_pos

        color = drone_colors[d_idx % len(drone_colors)]

        if path:
            xs = [p[0] + 0.5 for p in path]
            ys = [p[1] + 0.5 for p in path]

            ax2.plot(
                xs, ys,
                color=color,
                linewidth=2,
                label=drone_name
            )

        ax2.scatter(
            start_pos[0] + 0.5,
            start_pos[1] + 0.5,
            color=color,
            marker="o",
            s=70
        )

    ax2.set_title("Evolved UAV Flight Paths", fontweight="bold")
    ax2.legend()

    # 3. Sensor Coverage Heatmap
    heatmap = np.zeros(
        (env.grid_size, env.grid_size),
        dtype=float
    )

    for d_idx in range(ga.num_drones):
        mission = best_chromosome.get_drone_mission(d_idx)

        for wp in mission:
            if wp["sensor_active"] != 1:
                continue

            x_center = float(wp["x"])
            y_center = float(wp["y"])
            sensor_power = float(wp["sensor_power"])

            # Keep this consistent with the fitness function
            radius = 1.0 + 2.0 * sensor_power

            min_x = max(0, math.floor(x_center - radius))
            max_x = min(
                env.grid_size - 1,
                math.ceil(x_center + radius)
            )
            min_y = max(0, math.floor(y_center - radius))
            max_y = min(
                env.grid_size - 1,
                math.ceil(y_center + radius)
            )

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    distance_squared = (
                        (x - x_center) ** 2
                        + (y - y_center) ** 2
                    )

                    if distance_squared <= radius ** 2:
                        heatmap[y, x] += sensor_power

    draw_environment(ax3, alpha=0.25)

    masked_heatmap = np.ma.masked_where(
        heatmap == 0,
        heatmap
    )

    im = ax3.imshow(
        masked_heatmap,
        cmap="hot",
        origin="lower",
        extent=[0, env.grid_size, 0, env.grid_size],
        interpolation="nearest",
        alpha=0.9
    )

    # Mark priority zones
    priority_y, priority_x = np.where(env.priority_zones == 1)

    if len(priority_x) > 0:
        ax3.scatter(
            priority_x + 0.5,
            priority_y + 0.5,
            s=5,
            facecolors="none",
            edgecolors="cyan",
            label="Priority Zones"
        )
        ax3.legend(loc="upper right")

    ax3.set_title("Sensor Coverage Heatmap", fontweight="bold")

    fig.colorbar(
        im,
        ax=ax3,
        label="Accumulated Sensor Power"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Generating Visualization Graphs (Part 16)...")
    run_visualization()
