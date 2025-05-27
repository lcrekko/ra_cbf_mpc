"""
This provides many different plotters

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"] # Set the font to Times New Roman or similar
})

def plotter_kernel(ax: maxes.Axes, x_data: np.ndarray, y_data: np.ndarray,
                    info_text: dict, info_color: tuple,
                    marker=False) -> None:
    """
    This is the kernel of the main plotting function MonteCarloPlotter
    :param ax: the handle of the subplots, e.g., ax[1, 0] (the second row, first column)
    :param x_data: the x-axis data (1-D array)
    :param y_data: a bunch of y-axis data (2-D array)
    :param info_text: the text information, a dictionary that contains labels and titles
                1) "x_label": the text of the x-axis
                2) "y_label": the text of the y-axis
                3) "legend": legend information, may not be used at all
    :param info_color: tuple, color information, just a color
    :param marker: whether to show the marker
    Remark: this function will be frequently used in the class MonteCarloPlotter()
    """
    # Compute the max, min, mena and variance
    y_max = np.max(y_data, axis=0)
    y_min = np.min(y_data, axis=0)
    y_mean = np.mean(y_data, axis=0)
    # y_std = np.std(y_data, axis=0)

    # Creat color variations
    # color_bound = tuple(x * 0.75 for x in info_color)
    # color_variance = tuple(x * 0.5 for x in info_color)
    color_range = tuple(x * 0.25 for x in info_color)

    # basic mean plot
    if marker:
        color_marker_face = (info_color[0] * 0.75, np.min([1, info_color[1] * 1.25]), info_color[2])
        color_marker_edge = (np.min([1, info_color[0] * 1.25]), info_color[1] * 0.75, info_color[2])
        ax.plot(x_data, y_mean,
                label=info_text["legend"],
                linewidth=2.5, color=info_color,
                marker='.', markersize=10, markerfacecolor=color_marker_face,
                markeredgewidth=2, markeredgecolor=color_marker_edge)
    else:
        ax.plot(x_data, y_mean,
                label=info_text["legend"],
                linewidth=2.5, color=info_color)

    # plot the max and the min (fill the shaded color)
    ax.fill_between(x_data, y_min, y_max, color=color_range, alpha=0.25, label='Min-Max Envelope')

class RegretPlotter:
    def __init__(self, ax: maxes.Axes,
                 x_data: np.ndarray, y_data:np.ndarray,
                 info_text: dict, info_font: dict, info_color: tuple,
                 marker=False):
        """
        Initialize the basic plot information

        Parameters:
            ax: the handle of the subplots or a plot
            x_data: the x-axis data, 1-D array
            y_data: the y-axis data, 2-D array
            info_text: the text info of the plot, a dictionary, with the following information
                1) "x_label": the text of the x-axis
                2) "y_label": the text of the y-axis
                3) "legend": legend information, may not be used at all
            info_font: the font info of the plot, a dictionary, with the following information
                1) "ft_type": the type of the font
                2) "ft_size_label": label size
                3) "ft_size_legend": legend size
                4) "ft_size_tick": tick size
            info_color: a tuple, the theme color of the plot
            marker: BOOL, no marker is added by default
        """
        # Pass the information
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        self.info_text = info_text
        self.info_font = info_font
        self.info_color = info_color
        self.marker = marker

    def plot_unified(self, x_scale_log=False, y_scale_log=False, set_x_ticks=False):
        """
        This function is the basic plot, without any zoom in.

        Parameter:
            x_scale_log: whether the x_scale use the log
            y_scale_log: whether the y_scale use the log
            set_x_ticks: whether we actively control the ticks
        """
        # ----------- Plotting Section -----------
        plotter_kernel(self.ax,
                       self.x_data, self.y_data,
                       self.info_text, self.info_color,
                       self.marker)

        # ----------- Post-configuration -----------
        # set the title and the labels
        self.ax.set_xlabel(self.info_text["x_label"],
                            fontdict={'family': self.info_font["ft_type"],
                                      'size': self.info_font["ft_size_label"],
                                      'weight': 'bold'})
        self.ax.set_ylabel(self.info_text["y_label"],
                            fontdict={'family': self.info_font["ft_type"],
                                      'size': self.info_font["ft_size_label"],
                                      'weight': 'bold'})
        # set the x-axis
        if set_x_ticks:
            self.ax.set_xticks(self.x_data)
            # # Manually set tick labels as 1, 2, ..., 10
            # self.ax.set_xticklabels([str(i) for i in range(1, 11)])

            # # Hide default offset text
            # self.ax.xaxis.get_offset_text().set_visible(False)

            #     # Add the "× 10⁻⁴" manually below the axis
            # self.ax.annotate(r"$\times 10^{-3}$",
            #         xy=(1, -0.05),
            #         xycoords='axes fraction',
            #         ha='center', fontsize=20)
        # ax.set_ylabel('Y Label')

        # Use ScalarFormatter and disable offset notation
        # formatter = mticker.ScalarFormatter(useOffset=False)
        # self.ax.yaxis.set_major_formatter(formatter)

        # set size of the ticks
        self.ax.tick_params(axis='x', labelsize=self.info_font["ft_size_tick"])
        self.ax.tick_params(axis='y', labelsize=self.info_font["ft_size_tick"])

        # set the log-scale
        if x_scale_log:
            self.ax.set_xscale('log')
        if y_scale_log:
            self.ax.set_yscale('log')

        # set the background and grid
        self.ax.set_facecolor((0.95, 0.95, 0.95))
        self.ax.grid(True, linestyle='--', color='white', linewidth=1)
        
