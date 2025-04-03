import numpy as np
import matplotlib.pyplot as plt
import json

def plot_radial_benchmark(task_scores, upper_bound=1.0, task_name = "test", dark_mode=False):
    """
    Creates a radial polygonal benchmark plot with radial grid lines.
    
    Parameters:
    - task_scores: dict, keys are task names, values are scores.
    - upper_bound: float, maximum value for normalization.
    - dark_mode: bool, if True, sets background to black and text to white.
    """
    # Extract tasks and scores
    labels = list(task_scores.keys())
    scores = np.array(list(task_scores.values()))
    
    # Normalize scores to the range [0, 1]
    normalized_scores = scores / upper_bound
    
    # Number of variables
    num_vars = len(labels)
    
    # Compute angle for each task
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Close the scores loop
    normalized_scores = np.concatenate((normalized_scores, [normalized_scores[0]]))
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    
    # Set dark mode if specified
    if dark_mode:
        plt.style.use('dark_background')
        
        # Make ALL text white, with specific focus on axis elements
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['ytick.labelcolor'] = 'white'  # Specifically target y-tick labels
        plt.rcParams['xtick.labelcolor'] = 'white'  # Specifically target x-tick labels
        plt.rcParams['axes.edgecolor'] = 'white'
        # plt.rcParams['axes.titlecolor'] = 'white'
        # plt.rcParams['legend.facecolor'] = 'black'
        # plt.rcParams['legend.edgecolor'] = 'white'
        # plt.rcParams['legend.labelcolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['figure.edgecolor'] = 'black'
        plt.rcParams['grid.color'] = 'white'
        plt.rcParams['lines.color'] = 'white'
        plt.rcParams['patch.edgecolor'] = 'white'

        for spine in plt.gca().spines.values():
            spine.set_color('white')
        
        # Set the current axes background to black
        plt.gca().set_facecolor('black')
        
        # Additional direct method to ensure y-axis text is white
        for label in plt.gca().get_yticklabels():
            label.set_color('white')
        
        # Same for x-axis just to be thorough
        for label in plt.gca().get_xticklabels():
            label.set_color('white')
        
        task_name += "-dark"

        
    # Draw the polygon
    ax.fill(angles, normalized_scores, color='steelblue', alpha=0.3)
    ax.plot(angles, normalized_scores, color='steelblue', linewidth=2, linestyle='solid')
    
    # Add labels at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([]) 
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 1.2, label, fontsize=12, fontweight='bold', ha='center', va='center', transform=ax.transData)
    
    # Radial grid markers
    yticks = np.linspace(0, 1, 5)  # Adjust number of radial markers as needed
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y*upper_bound:.2f}" for y in yticks], fontsize=10)
    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.7)
    ax.xaxis.grid(True, linestyle='dashed', linewidth=0.7)

    
    # Add radial grid lines
    # ax.grid(True, which='both', linestyle='dashed', linewidth=0.7)

    plt.tight_layout()

    # Show the plot
    plt.savefig(f"outputs/{task_name}-radial-plot.png", dpi = 400)


def plot_radial_benchmark_multiple(task_scores_1, task_scores_2, upper_bound_1=1.0, upper_bound_2=1.0, labels=None, task_name = "test"):
    """
    Creates a radial polygonal benchmark plot for two metrics on different scales.
    
    Parameters:
    - task_scores_1: dict, first metric (keys are task names, values are scores).
    - task_scores_2: dict, second metric (keys are task names, values are scores).
    - upper_bound_1: float, maximum value for normalization of first metric.
    - upper_bound_2: float, maximum value for normalization of second metric.
    - labels: list, custom task names order (optional, defaults to task_scores_1 keys).
    """
    # Extract tasks and scores
    if labels is None:
        labels = list(task_scores_1.keys())
    
    scores_1 = np.array([task_scores_1[label] for label in labels])
    scores_2 = np.array([task_scores_2[label] for label in labels])
    
    # Normalize scores
    normalized_scores_1 = scores_1 / upper_bound_1
    normalized_scores_2 = scores_2 / upper_bound_2
    
    # Number of variables
    num_vars = len(labels)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Close the scores loop
    normalized_scores_1 = np.concatenate((normalized_scores_1, [normalized_scores_1[0]]))
    normalized_scores_2 = np.concatenate((normalized_scores_2, [normalized_scores_2[0]]))
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    
    # Draw polygons
    ax.fill(angles, normalized_scores_1, color='steelblue', alpha=0.3, label='Metric 1')
    ax.plot(angles, normalized_scores_1, color='steelblue', linewidth=2, linestyle='solid')
    
    ax.fill(angles, normalized_scores_2, color='lightcoral', alpha=0.3, label='Metric 2')
    ax.plot(angles, normalized_scores_2, color='lightcoral', linewidth=2, linestyle='solid')
    
    # Add labels at the correct angles with increased radial distance
    ax.set_xticklabels([]) 
    ax.set_xticks(angles[:-1])
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 1.15, label, fontsize=12, fontweight='bold', ha='center', va='center', transform=ax.transData)
    
    # Radial grid markers
    yticks = np.linspace(0, 1, 5)  # Adjust number of radial markers as needed
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y*upper_bound_1:.2f}" for y in yticks], fontsize=10)
    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.7)
    
    # Title and legend
    
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"outputs/{task_name}-radial-plot-multiple.png", dpi = 400)
    

# Example usage
# task_scores = {
#     "Task A": 0.8,
#     "Task B": 0.6,
#     "Task C": 0.9,
#     "Task D": 0.75,
#     "Task E": 0.85
# }
# plot_radial_benchmark(task_scores, upper_bound=1.0)

# animals, vehicles


shard_name = "sd1.4-base"


with open(f'results/{shard_name}.json', 'r') as f:
    scores1 = json.load(f)

with open(f'results/sd1.4-base.json', 'r') as f:
    scores2 = json.load(f)


# tasks = ["animals", "body-parts", "clothes", "electronics", "text", "vehicles"]

task_scores1 = {}
task_scores2 = {}

for key in list(scores1.keys()):
    task_scores1[key] = scores1[key]["ir"]
    task_scores2[key] = scores2[key]["ir"]
    # task_scores2[key] = scores[task][key]["clip"]




plot_radial_benchmark(task_scores1, upper_bound= 0.3, task_name = shard_name, dark_mode=True)


# quit()

# task_scores_1 = {
#     "Task A": 0.8,
#     "Task B": 0.6,
#     "Task C": 0.9,
#     "Task D": 0.75,
#     "Task E": 0.85
# }

# task_scores_2 = {
#     "Task A": 50,
#     "Task B": 85,
#     "Task C": 70,
#     "Task D": 55,
#     "Task E": 100
# }

# plot_radial_benchmark_multiple(task_scores1, task_scores2, upper_bound_1=1.1 * max(task_scores1.values()), upper_bound_2=1.1 * max(task_scores2.values()), task_name = task)
# 