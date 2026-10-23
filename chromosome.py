
import numpy as np
from environment import DisasterEnvironment


class Chromosome:
    def __init__(self, environment, num_drones=3, waypoints_per_drone=5):
        self.env = environment
        self.num_drones = num_drones
        self.waypoints_per_drone = waypoints_per_drone

        # Each waypoint contains x, y, speed, sensor power, sensor active
        self.genes_per_wp = 5
        self.total_genes = (
            self.num_drones * self.waypoints_per_drone * self.genes_per_wp
        )

        if self.num_drones > len(self.env.drone_starts):
            raise ValueError("Not enough drone starting positions.")

        self.genes = np.zeros(self.total_genes, dtype=float)
        self.fitness = 0.0

        self._initialize_random_genes()

    def _initialize_random_genes(self):
        for drone_idx in range(self.num_drones):

            drone_name = list(self.env.drone_starts.keys())[drone_idx]
            start_position = self.env.drone_starts[drone_name]

            for wp_idx in range(self.waypoints_per_drone):

                start_idx = (
                    (drone_idx * self.waypoints_per_drone + wp_idx)
                    * self.genes_per_wp
                )

                # First waypoint is the drone's assigned starting position
                if wp_idx == 0:
                    x, y = start_position
                else:
                    x, y = self._get_random_valid_coordinate()

                speed = np.random.uniform(1.0, 5.0)
                sensor_power = np.random.uniform(0.1, 1.0)
                sensor_active = np.random.choice([0, 1])

                self.genes[start_idx] = x
                self.genes[start_idx + 1] = y
                self.genes[start_idx + 2] = speed
                self.genes[start_idx + 3] = sensor_power
                self.genes[start_idx + 4] = sensor_active

    def _get_random_valid_coordinate(self):
        while True:
            x = np.random.randint(0, self.env.grid_size)
            y = np.random.randint(0, self.env.grid_size)

            if self.env.is_valid_position(x, y):
                return x, y

    def get_drone_mission(self, drone_idx):
        if not 0 <= drone_idx < self.num_drones:
            raise ValueError("Invalid drone index.")

        start_idx = (
            drone_idx * self.waypoints_per_drone * self.genes_per_wp
        )
        end_idx = start_idx + (
            self.waypoints_per_drone * self.genes_per_wp
        )

        drone_genes = self.genes[start_idx:end_idx]
        mission = []

        for wp_idx in range(self.waypoints_per_drone):
            wp_start = wp_idx * self.genes_per_wp

            waypoint = {
                "x": int(drone_genes[wp_start]),
                "y": int(drone_genes[wp_start + 1]),
                "speed": round(drone_genes[wp_start + 2], 2),
                "sensor_power": round(drone_genes[wp_start + 3], 2),
                "sensor_active": int(drone_genes[wp_start + 4])
            }

            mission.append(waypoint)

        return mission

    def __str__(self):
        return (
            f"Chromosome(Total Genes: {self.total_genes}, "
            f"Fitness: {self.fitness})"
        )


if __name__ == "__main__":
    environment = DisasterEnvironment()
    chromosome = Chromosome(environment)

    print("Chromosome generated successfully!")
    print(chromosome)
    print("Total genes:", len(chromosome.genes))

    for drone_idx in range(chromosome.num_drones):
        print(f"\nDrone {drone_idx + 1} Mission:")

        mission = chromosome.get_drone_mission(drone_idx)

        for i, waypoint in enumerate(mission):
            print(f"Waypoint {i + 1}: {waypoint}")
