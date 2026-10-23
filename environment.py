
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


class DisasterEnvironment:

    def __init__(self, grid_size=50):
        self.grid_size = grid_size

        # Main grid and separate zone masks
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.no_fly_zones = np.zeros((grid_size, grid_size), dtype=int)
        self.hazard_zones = np.zeros((grid_size, grid_size), dtype=int)
        self.priority_zones = np.zeros((grid_size, grid_size), dtype=int)

        # Drone starting positions: (x, y)
        self.drone_starts = {
            "Drone 1": (2, 2),
            "Drone 2": (47, 2),
            "Drone 3": (25, 47)
        }

        self.create_zones()
        self.update_grid()
        self.validate_drone_starts()

    def create_zones(self):

        # No-fly zones
        self.no_fly_zones[0:5, 20:30] = 1
        self.no_fly_zones[25:30, 20:30] = 1
        self.no_fly_zones[35:50, 15:18] = 1

        # Environmental hazard zones
        self.hazard_zones[20:28, 5:12] = 1
        self.hazard_zones[15:20, 30:40] = 1
        self.hazard_zones[40:48, 40:48] = 1

        # High-priority disaster zones
        self.priority_zones[10:15, 10:15] = 1
        self.priority_zones[30:38, 25:35] = 1
        self.priority_zones[5:10, 40:45] = 1

    def update_grid(self):
        self.grid.fill(0)

        self.grid[self.priority_zones == 1] = 3
        self.grid[self.hazard_zones == 1] = 2
        self.grid[self.no_fly_zones == 1] = 1
        
    def is_valid_position(self, x, y):

        # Check grid boundaries
        if not (0 <= x < self.grid_size and
                0 <= y < self.grid_size):
            return False

        # NumPy indexing is [row, column] = [y, x]
        return self.no_fly_zones[y, x] == 0

    def validate_drone_starts(self):

        for name, (x, y) in self.drone_starts.items():
            if not self.is_valid_position(x, y):
                raise ValueError(
                    f"{name} has an invalid starting position: {(x, y)}"
                )

    def display_environment(self):

        colors = ["#E8E8E8", "#1A1A1A", "#FF4D4D", "#33CC33"]
        cmap = ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(10, 9))

        ax.imshow(
            self.grid,
            cmap=cmap,
            origin="lower",
            extent=[0, self.grid_size, 0, self.grid_size],
            vmin=0,
            vmax=3
        )

        # Show drone starting positions
        for name, (x, y) in self.drone_starts.items():
            ax.scatter(
                x + 0.5, y + 0.5,
                marker="^",
                s=150,
                color="blue",
                edgecolors="white"
            )

            ax.text(
                x + 0.7, y + 0.7,
                name,
                fontsize=8,
                color="blue",
                fontweight="bold"
            )

        ax.set_xticks(np.arange(0, self.grid_size + 1, 5))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 5))
        ax.grid(color="white", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("UAV Disaster Environment (50 x 50)")

        legend_items = [
            Patch(facecolor="#E8E8E8", edgecolor="black",
                  label="Safe Cell (0)"),
            Patch(facecolor="#1A1A1A",
                  label="No-Fly Zone (1)"),
            Patch(facecolor="#FF4D4D",
                  label="Hazard Zone (2)"),
            Patch(facecolor="#33CC33",
                  label="Priority Zone (3)"),
            plt.Line2D(
                [0], [0],
                marker="^",
                color="w",
                markerfacecolor="blue",
                markersize=10,
                label="Drone Start"
            )
        ]

        ax.legend(
            handles=legend_items,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.tight_layout()
        plt.show()


# Create and display the environment
if __name__ == "__main__":

    environment = DisasterEnvironment()
    environment.display_environment()

    print("Grid size:", environment.grid.shape)
    print("No-fly cells:", np.sum(environment.no_fly_zones))
    print("Hazard cells:", np.sum(environment.hazard_zones))
    print("Priority cells:", np.sum(environment.priority_zones))
    print("Drone positions:", environment.drone_starts)

    print("\nStarting position validation:")
    for name, (x, y) in environment.drone_starts.items():
        print(name, (x, y), "Valid:", environment.is_valid_position(x, y))
