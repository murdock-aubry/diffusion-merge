import matplotlib.pyplot as plt


def scatter_plot(x_data, y_data, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis', path_name = "outputs/image.png"):
    """
    Create a scatter plot with a black background and white axes.
    
    Parameters:
    x_data (array-like): Data for x-axis
    y_data (array-like): Data for y-axis
    title (str, optional): Plot title
    xlabel (str, optional): X-axis label
    ylabel (str, optional): Y-axis label
    """
    # Set up the plot with a black background
    plt.figure(figsize=(10, 6), facecolor='black')
    plt.gcf().patch.set_facecolor('black')
    
    # Create the scatter plot with larger, purple, hollow circles
    plt.scatter(x_data, y_data, color='purple', marker='o', edgecolors='purple', s=100, linewidths=5, facecolors='none')
    
    # Customize the plot appearance
    plt.title(title, color='white', fontsize=18)  # Increased font size for title
    plt.xlabel(xlabel, color='white', fontsize=16)  # Increased font size for x-axis label
    plt.ylabel(ylabel, color='white', fontsize=16)  # Increased font size for y-axis label
    
    # Set background color
    plt.gca().set_facecolor('black')
    
    # Customize axes colors
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    
    # Customize tick colors
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    
    # Show the plot
    plt.savefig(path_name, dpi = 400)