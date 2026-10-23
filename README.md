# UAV Disaster Management Strategy Engine Using Genetic Algorithm

## 1. Project Overview

The **UAV Disaster Management Strategy Engine** is a Python-based simulation that uses a Genetic Algorithm (GA) to develop UAV (Unmanned Aerial Vehicle) mission strategies for disaster-management and autonomous-surveillance scenarios.

The environment is represented as a 50 × 50 grid containing priority zones, hazard zones, and no-fly zones. Three UAVs are assigned missions consisting of multiple waypoints, movement speeds, and sensor configurations.

The Genetic Algorithm evolves these missions over multiple generations to improve priority-zone coverage and target detection while reducing energy consumption, flight time, and hazard exposure. A* pathfinding is used to find routes between waypoints while accounting for obstacles.

The project also includes visualization, random-search comparison, and experiments to study the effects of different Genetic Algorithm parameters.

## 2. Project Objectives

- Develop an object-oriented simulation environment for UAV mission planning.
- Represent UAV missions using a custom chromosome structure.
- Generate and evolve a population of candidate UAV strategies.
- Design a fitness function that balances coverage, detection, energy, time, and risk.
- Implement selection, custom crossover, mutation, and elitism.
- Handle constraints using chromosome repair and fitness penalties.
- Visualize evolved routes and sensor coverage.
- Compare Genetic Algorithm performance against random search.
- Investigate the effects of population size, mutation rate, and selection method.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Grid and numerical operations |
| Matplotlib | Environment, route, heatmap, and experiment visualizations |
| Visual Studio Code | Development and execution environment |
| Git and GitHub | Version control and project submission |

The implementation also uses Python standard-library modules, including `math`, `heapq`, and `copy`.

## 4. Repository Structure

```text
UAV-Disaster-Management-GA/
│
├── environment.py
├── chromosome.py
├── genetic_algorithm.py
├── visualization.py
├── experiments.py
├── requirements.txt
└── README.md
```

### File descriptions

| File | Responsibility |
|---|---|
| `environment.py` | Creates the grid, UAV starting positions, priority zones, hazard zones, and no-fly zones. |
| `chromosome.py` | Defines the chromosome structure and generates candidate UAV missions. |
| `genetic_algorithm.py` | Implements fitness evaluation, A* pathfinding, selection, crossover, mutation, repair, elitism, and the GA loop. |
| `visualization.py` | Visualizes fitness progress, evolved UAV paths, and sensor coverage. |
| `experiments.py` | Compares GA with random search and evaluates selected hyperparameter configurations. |
| `requirements.txt` | Lists the external Python libraries required to run the project. |
| `README.md` | Documents the project, installation, execution, and implementation. |

**Note:** `visualization.py` and `experiments.py` are descriptive filenames. If your actual Part 16 or Parts 17–18 scripts have different names, use those filenames in the repository and update the execution commands accordingly.

## 5. Environment Design

The simulation uses a 50 × 50 grid. Each cell represents a location that a UAV may visit or encounter during its mission.

The environment contains:

- **Priority zones:** Areas that receive greater attention because they represent important locations for disaster assessment or surveillance.
- **Hazard zones:** Areas associated with additional risk during UAV movement.
- **No-fly zones:** Restricted areas that UAV routes must avoid.
- **UAV starting positions:** Three predefined starting locations, one for each UAV.

The environment module provides position validation and visualization functions. No-fly restrictions are considered during pathfinding and chromosome repair, while hazard exposure contributes to the fitness calculation.

## 6. Chromosome Representation

A custom chromosome represents the mission strategy for all three UAVs.

Each waypoint contains five genes:

1. **X-coordinate:** Horizontal grid position.
2. **Y-coordinate:** Vertical grid position.
3. **Speed:** UAV movement speed.
4. **Sensor power:** Sensor intensity or power level.
5. **Sensor activation:** Indicates whether the sensor is active.

The default configuration uses three UAVs with five waypoints per UAV.

Therefore, the chromosome contains:

\[
3 \times 5 \times 5 = 75 \text{ genes}
\]

The first waypoint of each UAV is assigned to its predefined starting position. The remaining waypoint coordinates and mission parameters are initialized within their configured ranges.

This representation allows the algorithm to evolve both UAV routes and sensor configurations.

## 7. Genetic Algorithm Implementation

The project implements the following Genetic Algorithm workflow.

### Part 1 — Environment initialization

Creates the grid and defines priority, hazard, and no-fly zones.

### Part 2 — Chromosome representation

Encodes waypoint coordinates, speed, sensor power, and sensor activation for each UAV.

### Part 3 — Initial population

Generates a population of candidate UAV missions using randomized chromosome initialization.

### Part 4 — Constraint handling

Uses coordinate validation, chromosome repair, and fitness penalties to handle invalid or unsuitable missions. Initial-population repair is explicitly performed in the experiments script.

### Part 5 — Fitness function

Evaluates each candidate mission using coverage and detection rewards, together with penalties for resource consumption and risk.

The implemented weighted fitness function is:

\[
\begin{aligned}
F ={}& 10C + 10D - 2E - T - 5H\\
&-1000N - 500R
\end{aligned}
\]

Where:

- \(C\) = priority-zone coverage rate.
- \(D\) = target detection rate.
- \(E\) = estimated energy consumption.
- \(T\) = estimated flight time.
- \(H\) = hazard exposure.
- \(N\) = no-fly-zone or invalid-path violation count.
- \(R\) = flight-range violation count.

Coverage and detection rates are represented as percentages in the implementation. The fitness weights determine the relative importance of each objective.

The algorithm uses a weighted fitness score rather than a Pareto-based multi-objective optimization method.

### Part 6 — Roulette-wheel selection

Selects parent chromosomes according to their adjusted fitness values. Fitness adjustment allows selection to operate when candidate fitness values are negative.

### Part 7 — Custom crossover

Combines waypoint genes from two parent chromosomes to generate offspring. The first waypoint is preserved from the first parent, while the remaining waypoint data is selected from the parents.

### Part 8 — Offspring constraint handling

Applies chromosome repair to offspring to correct invalid coordinates and attempt to improve route feasibility.

### Part 9 — Custom mutation

Introduces random changes to waypoint coordinates, speed, sensor power, or sensor activation. Coordinate mutations generate new valid coordinate pairs.

### Part 10 — Mutation constraint handling

Repairs mutated chromosomes and resets their stored fitness and metrics so they can be evaluated again.

### Part 11 — Elitism

Preserves the two highest-ranked chromosomes when generating the next population, helping retain strong candidate solutions.

### Part 12 — New population generation

Combines preserved elite chromosomes with newly generated offspring to form the next generation.

### Part 13 — Generation counter

Tracks progress through successive generations.

### Part 14 — Termination criteria

Stops the main GA loop when the configured generation limit is reached. The experiments script also uses a fitness-evaluation budget for fair comparisons.

### Part 15 — Complete Genetic Algorithm

Combines population initialization, fitness evaluation, selection, crossover, mutation, constraint repair, elitism, and termination into the complete optimization workflow.

## 8. A* Pathfinding and Constraint Handling

A* pathfinding is used to calculate routes between consecutive UAV waypoints.

The implementation uses:

- A four-directional grid movement model.
- Manhattan distance as the heuristic.
- No-fly-zone checks during route search.
- Route-distance calculations for flight-range evaluation.

Chromosome repair attempts to correct invalid coordinates and unreachable waypoint connections. The fitness function also applies substantial penalties for constraint violations.

**Important limitation:** Repair is not guaranteed to produce a feasible route in every case. Flight-range compliance and no-fly-zone violations should therefore be checked in the reported metrics rather than assumed.

## 9. Visualization — Part 16

The visualization script generates three main plots.

### Fitness progress

Displays the best-so-far fitness and average population fitness across generations. These curves help illustrate the optimization process.

### Evolved UAV flight paths

Displays the environment and the routes connecting each UAV's waypoints using A* pathfinding.

### Sensor coverage heatmap

Displays the grid cells covered by active UAV sensors. Sensor coverage depends on waypoint locations, sensor activation, and sensor power.

These visualizations help assess the quality and behavior of the evolved missions.

## 10. Random Search Comparison — Part 17

The project compares the Genetic Algorithm with a random-search baseline using a shared fitness-evaluation budget of 400 evaluations.

The random-search baseline generates and repairs candidate chromosomes, evaluates their fitness, and tracks the best solution found so far.

The comparison plots best-so-far fitness against the number of fitness evaluations.

Using the same evaluation budget makes the comparison more meaningful because both approaches receive the same number of fitness evaluations. Results can still vary due to randomness.

## 11. Hyperparameter Experiments — Part 18

The experiments investigate how selected Genetic Algorithm parameters affect optimization performance. The configured evaluation budget is 600 evaluations per experiment.

| Parameter | Configurations |
|---|---|
| Population size | 10, 20, and 30 |
| Mutation rate | 0.05, 0.15, and 0.40 |
| Selection method | Roulette-wheel selection and tournament selection |
| Tournament size | 2 and 5 |

The experiments plot best-so-far fitness against the number of evaluations.

These comparisons help investigate:

- How population size affects exploration and convergence.
- How mutation rate affects population diversity.
- How tournament size changes selection pressure.
- How different selection strategies perform under the same evaluation budget.

The experiments provide empirical comparisons; they do not guarantee that one parameter configuration will always be superior.

## 12. Installation and Setup

### Prerequisites

- Python 3 installed on your system.
- Visual Studio Code or another Python-compatible editor.
- Internet access for the initial installation of dependencies.

### Step 1: Open the project

Open the `UAV-Disaster-Management-GA` folder in Visual Studio Code.

### Step 2: Install dependencies

Open the VS Code terminal and run:

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
matplotlib
```

### Step 3: Verify the installation

Run:

```bash
python -c "import numpy; import matplotlib; print('Dependencies installed successfully')"
```

## 13. How to Run the Project

Run the commands below from the project directory.

### Environment

```bash
python environment.py
```

### Chromosome representation

```bash
python chromosome.py
```

### Complete Genetic Algorithm

```bash
python genetic_algorithm.py
```

### Visualization

```bash
python visualization.py
```

### Random search and hyperparameter experiments

```bash
python experiments.py
```

The visualization and experiment scripts may take longer because they perform repeated fitness evaluations and generate plots.

If your script filenames differ, replace the names in these commands with the actual filenames in your repository.

## 14. Expected Outputs

Depending on the script being executed, the project produces:

- A visualization of the disaster environment.
- Chromosome and Genetic Algorithm test output.
- Best and average fitness curves.
- Evolved UAV flight paths.
- A sensor coverage heatmap.
- A Genetic Algorithm versus random-search comparison.
- Hyperparameter comparison plots.

The exact numerical results depend on the random initialization and evolution process.

## 15. Limitations and Future Improvements

The current implementation provides a simplified simulation of UAV mission planning. Potential improvements include:

- Stronger guarantees for flight-range feasibility.
- More advanced collision avoidance and dynamic obstacle handling.
- More realistic UAV energy and flight-time models.
- Repeated trials with random seeds and statistical summaries.
- Additional sensor models and more realistic detection probabilities.
- More advanced multi-objective optimization techniques.

## 16. Conclusion

This project demonstrates how a Genetic Algorithm can be applied to UAV mission planning in a simulated disaster-management environment. It integrates object-oriented programming, chromosome design, heuristic pathfinding, fitness evaluation, evolutionary operators, constraint handling, visualization, and experimental comparison.

The implementation also provides a foundation for studying how optimization parameters influence the quality of evolved UAV strategies.

