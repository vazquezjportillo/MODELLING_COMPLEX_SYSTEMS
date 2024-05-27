
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button, TextBox
import numpy as np


class LatticeGUI:
    """
        GUI for displaying lattice in color
    """

    def __init__(self, lattice):
        """ Configure window """
        self.lattice = lattice
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(right=0.65)
        plt.title("Bio-LGCA\nstochastic simulation", fontsize=14, fontweight='bold')

        # Generate infection map
        self.plot = self.create_honeycomb()
        self.plot.set_facecolor(self.get_lattice_colors())
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Add bar chart with counts
        self.state_count_ax = self.fig.add_axes([0.75, 0.7, 0.2, 0.2])
        self.state_counts = count_states(self.lattice)
        self.bars = self.state_count_ax.bar(['S', 'I', 'R', 'V'], self.state_counts, color=['#6481eb', '#e84d8a', '#4de89e', '#64c5eb'])
        self.state_count_ax.set_ylim(0, N**2 * 5)
        self.state_count_ax.set_yticks([0, N**2 * 4])
        
        # Widgets
        self.iteration = 0
        self.iteration_display = self.fig.text(0.79, 0.6, f'Iteration: {self.iteration}', fontsize=10)

        self.administrations_ax = plt.axes([0.78, 0.45, 0.15, 0.05])
        self.administrations = TextBox(self.administrations_ax, '', initial='100', textalignment='center')

        self.vaccinate_button_ax = plt.axes([0.78, 0.4, 0.15, 0.05])
        self.vaccinate_button = Button(self.vaccinate_button_ax, 'Vaccinate')
        self.vaccinate_button.on_clicked(self.vaccinate)

        self.update_button_ax = plt.axes([0.78, 0.1, 0.15, 0.05])
        self.update_button = Button(self.update_button_ax, 'Update')
        self.update_button.on_clicked(self.update_state)

        self.save_button_ax = plt.axes([0.78, 0.05, 0.15, 0.05])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.save_image)

        plt.show()

    def create_honeycomb(self):
        """ Create and return a plot with hexagonal cells """
        radius = 0.5
        hexagons = []
        for (i, j), _ in np.ndenumerate(self.lattice):
            x = j * 1.5 * radius
            y = i * np.sqrt(3) * radius + (j % 2) * np.sqrt(3) / 2 * radius
            hexagon = RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.pi/6, facecolor='white', edgecolor='k')
            hexagons.append(hexagon)
        hex_collection = PatchCollection(hexagons, match_original=True)
        self.ax.add_collection(hex_collection)
        self.ax.autoscale_view()
        return hex_collection
    
    def get_lattice_colors(self):
        """ Generate a list of colors based on the current state of each cell in the lattice """
        state_colors = {'S': np.array([100, 129, 235]) / 255,
                        'I': np.array([232, 77, 138]) / 255,
                        'R': np.array([77, 232, 158]) / 255,
                        'V': np.array([100, 197, 235]) / 255}
        colors = []
        for (i, j), cell in np.ndenumerate(self.lattice):
            total = 0
            color_sum = np.zeros(3)
            for particle in cell.particles:
                if particle:
                    color_sum += state_colors[particle.state]
                    total += 1
            if total > 0:
                # Use average state color
                color = color_sum / total
            else:
                # Empty cell is white
                color = np.array([1.0, 1.0, 1.0])
            colors.append(color)
        return colors

    def update_state(self, event):
        """ Call simulation functions and update state """
        self.iteration += 1
        self.iteration_display.set_text(f'Iteration: {self.iteration}')
        contact_operation(self.lattice)
        redistribute_operation(self.lattice)
        self.lattice = traversal_operation(self.lattice)
        self.update_display()

    def update_display(self):
        """ Update display """
        self.plot.set_facecolor(self.get_lattice_colors())
        self.state_counts = count_states(self.lattice)
        self.state_count_ax.cla()
        self.state_count_ax.set_ylim(0, N**2 * 5)
        self.state_count_ax.set_yticks([0, N**2 * 4])
        self.bars = self.state_count_ax.bar(['S', 'I', 'R', 'V'], self.state_counts, color=['#6481eb', '#e84d8a', '#4de89e', '#64c5eb'])
        self.fig.canvas.draw()

    def vaccinate(self, event):
        """ Move n people from 'S' to 'V' at random """
        try:
            n = int(self.administrations.text)
            susceptible = [particle for cell in self.lattice.flat for particle in cell.particles if particle and particle.state == 'S']
            np.random.shuffle(susceptible)
            for particle in susceptible[:n]:
                particle.state = 'V'
            self.update_display()
        except ValueError:
            print("Invalid number of administrations. Enter a positive integer.")

    def save_image(self, event):
        """ Save the current image of the lattice to a file """
        self.fig.savefig('Bio-LGCA.pdf')