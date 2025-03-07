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

def main():
    # Load the data (assuming it's saved to a file)
    with open('/w/284/murdock/merge/metrics.json', 'r') as f:
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
    plt.savefig('benchmark_plots.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()