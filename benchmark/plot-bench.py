import json
import matplotlib.pyplot as plt
import re
import numpy as np

def parse_model_name(model_name):
    # For a model name like "sdxl1_0.95_sdxl1-tuned_0.05", extract the first number (0.95)
    match = re.search(r'sdxl1_(\d+\.\d+)_', model_name)
    if match:
        return float(match.group(1))
    elif model_name == "sdxl1":
        return 1.0  # Base model is equivalent to 1.0
    return None

def plot_model_metrics(json_path, output_path='model_metrics.png'):
    """
    Plot model metrics as a grouped bar chart with separate scales and max score lines.
    
    Parameters:
    json_path (str): Path to the JSON file containing metrics
    output_path (str, optional): Path to save the output PNG file
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Prepare data for plotting
    models = list(data.keys())
    clip_scores = [data[model]['parti-prompts']['clip'] for model in models]
    ir_scores = [data[model]['parti-prompts']['ir'] for model in models]
    
    # Calculate max scores
    max_clip_score = max(clip_scores)
    max_ir_score = max(ir_scores)
    
    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Set the width of each bar and positions
    bar_width = 0.35
    index = np.arange(len(models))
    
    # Plot CLIP scores on the left y-axis
    color1 = '#3498db'
    rects1 = ax1.bar(index, clip_scores, bar_width, label='CLIP Score', color=color1, alpha=0.8)
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CLIP Score', color=color1, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(index + bar_width/2)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # Add horizontal dashed line for max CLIP score
    ax1.axhline(y=max_clip_score, color=color1, linestyle='--', alpha=0.5, 
                label=f'Max CLIP Score ({max_clip_score:.2f})')
    
    # Create a second y-axis for IR scores
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    rects2 = ax2.bar(index + bar_width, ir_scores, bar_width, label='IR Score', color=color2, alpha=0.8)
    ax2.set_ylabel('IR Score', color=color2, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add horizontal dashed line for max IR score
    ax2.axhline(y=max_ir_score, color=color2, linestyle='--', alpha=0.5, 
                label=f'Max IR Score ({max_ir_score:.2f})')
    
    # Title
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add grid to first axis
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot in high quality
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close()


def plot_polygon_bench():

    return

def main():
    # Load the data (assuming it's saved to a file)
    with open('/w/284/murdock/merge/benchmark/metrics.json', 'r') as f:
        data = json.load(f)
    
    # Extract the model proportions and corresponding benchmark values
    model_proportions = []
    clip_scores = []
    ir_scores = []
    
    for model_name, benchmarks in data.items():
        proportion = parse_model_name(model_name)
        if proportion is not None:
            model_proportions.append(proportion)
            clip_scores.append(benchmarks['parti-prompts']['clip'])
            
            # IR score might not be present for all models
            ir_score = benchmarks['parti-prompts'].get('ir')
            ir_scores.append(ir_score if ir_score is not None else np.nan)
    
    # Sort the data by model proportion
    sorted_indices = np.argsort(model_proportions)
    model_proportions = [model_proportions[i] for i in sorted_indices]
    clip_scores = [clip_scores[i] for i in sorted_indices]
    ir_scores = [ir_scores[i] for i in sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot CLIP scores
    plt.subplot(2, 1, 1)
    plt.plot(model_proportions, clip_scores, 'o-', color='blue', label='CLIP Score')
    plt.xlabel('First Model Proportion')
    plt.ylabel('CLIP Score')
    plt.title('CLIP Score vs. Model Proportion')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot IR scores
    plt.subplot(2, 1, 2)
    # Filter out any NaN values
    valid_indices = [i for i, ir in enumerate(ir_scores) if not np.isnan(ir)]
    valid_proportions = [model_proportions[i] for i in valid_indices]
    valid_ir_scores = [ir_scores[i] for i in valid_indices]
    
    plt.plot(valid_proportions, valid_ir_scores, 'o-', color='green', label='IR Score')
    plt.xlabel('First Model Proportion')
    plt.ylabel('IR Score')
    plt.title('IR Score vs. Model Proportion')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_plots_radial.png', dpi=300)
    plt.show()

if __name__ == "__main__":

    plot_model_metrics('metrics.json')
    main()