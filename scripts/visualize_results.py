"""
Visualization Module for Structure-Aware Chunking Results.

This script generates three key figures for the research paper/README:
1. Performance Comparison (Bar Chart): Baseline vs. Structure-Aware across domains.
2. Mechanism Visualization (Heatmap): Illustrating the penalty mechanism on a specific sample.
3. Parameter Sensitivity (Line Plot): Robustness analysis of the structure weight parameter.

Usage:
    python visualize_results.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add src module to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core import StructureAwareChunker

# Configuration
DATA_JSON_PATH = "../data/benchmark_50.json"
EVAL_CSV_PATH = "../data/eval_summary.csv"
IMG_OUTPUT_DIR = "../images"

# Matplotlib Style Configuration for Academic Publishing
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

warnings.filterwarnings("ignore")


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_performance_comparison():
    """
    Generates Figure 1: Domain-specific performance comparison (AMI Score).
    Reads aggregated data from eval_summary.csv.
    """
    print("Generating Figure 1: Performance Comparison...")
    
    if not os.path.exists(EVAL_CSV_PATH):
        print(f"Error: {EVAL_CSV_PATH} not found. Please run run_eval.py first.")
        return

    df = pd.read_csv(EVAL_CSV_PATH)
    
    # Setup canvas
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    
    # Plot bars
    # Using neutral gray for Baseline and scientific green for Ours
    rects1 = ax.bar(x - width/2, df['Base AMI'], width, label='Baseline (Semantic Only)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, df['Ours AMI'], width, label='Ours (Structure-Aware)', color='#27ae60')
    
    # Formatting
    ax.set_ylabel('Adjusted Mutual Information (AMI)')
    ax.set_title('Segmentation Performance by Domain (N=50)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Domain'])
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save
    save_path = os.path.join(IMG_OUTPUT_DIR, 'result_comparison.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_mechanism_heatmap():
    """
    Generates Figure 2: Heatmap visualization of the penalty mechanism.
    
    [Reference Logic]
    - Based on the user's provided snippet.
    - Baseline: Shows high similarity (Yellow box) between Header and previous sentence.
    - Ours: Shows reduced similarity (Green box) due to structure penalty.
    """
    print("Generating Figure 2: Mechanism Heatmap...")
    
    if not os.path.exists(DATA_JSON_PATH):
        print("Data file not found.")
        return

    with open(DATA_JSON_PATH, 'r', encoding='utf-8') as f:
        data_records = json.load(f)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

    # --- 1. Find the Perfect Visualization Sample ---
    # We look for a window where a sentence starts with "# Section" (Standard Markdown Header)
    window_sents = []
    header_idx = -1 # Relative index of the header in the window
    
    for record in data_records:
        sents = record['sentences']
        # Search range: avoid start and end of doc to ensure enough context
        for i in range(2, len(sents) - 5):
            if sents[i].startswith("# Section"):
                # Logic from user snippet: Window = 4 before + Header + 4 after
                window_sents = sents[i-4 : i+5]
                header_idx = 4 # The 5th sentence (index 4) is the header
                break
        if window_sents:
            break
            
    if not window_sents:
        print("No suitable header sample found for visualization.")
        return

    print(f"  Sample found: '{window_sents[header_idx]}'")

    # --- 2. Compute Matrices ---
    embs = model.encode(window_sents)
    sim_matrix = cosine_similarity(embs)
    
    # Simulate Ours: Manually apply penalty for visualization
    # This matches the user's logic: adj_matrix[h, k] -= penalty
    adj_matrix = sim_matrix.copy()
    penalty = 1.0 # Strong penalty for headers
    
    # Cut the connection between Header and ALL previous sentences in the window
    for k in range(header_idx):
        adj_matrix[header_idx, k] -= penalty
        adj_matrix[k, header_idx] -= penalty

    # --- 3. Plotting (Academic Style) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Custom Labels
    labels = [f"S{i}" for i in range(len(window_sents))]
    labels[header_idx] = "HEADER"

    # Plot A: Baseline (Problem)
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=ax1, cbar=False)
    ax1.set_title("(a) Baseline: Semantic Bleeding", fontsize=12, fontweight='bold', pad=12)
    
    # Highlight Box: Yellow (Warning)
    # Location: (3, 4) -> x=3 (Prev Sent), y=4 (Header)
    # Color: #f1c40f is "Academic Yellow" (more readable than pure 'yellow')
    rect1 = patches.Rectangle((3, 4), 1, 1, linewidth=3, edgecolor='#f1c40f', facecolor='none')
    ax1.add_patch(rect1)

    # Plot B: Ours (Solution)
    sns.heatmap(adj_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=ax2, cbar=True)
    ax2.set_title("(b) Ours: Structure Penalty Applied", fontsize=12, fontweight='bold', pad=12)
    
    # Highlight Box: Green (Correct/Safe)
    # Color: #2ecc71 is "Academic Green" (less neon than #00FF00)
    rect2 = patches.Rectangle((3, 4), 1, 1, linewidth=3, edgecolor='#2ecc71', facecolor='none')
    ax2.add_patch(rect2)

    plt.tight_layout()
    save_path = os.path.join(IMG_OUTPUT_DIR, 'heatmap_mechanism.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_sensitivity_analysis():
    """
    Generates Figure 3: Parameter Sensitivity Analysis.
    Evaluates AMI score stability across varying structure weights (lambda).
    """
    print("Generating Figure 3: Sensitivity Analysis (this may take a moment)...")
    
    # Load Data & Model
    with open(DATA_JSON_PATH, 'r') as f:
        # Use a subset (first 10 docs) for speed, or all for rigor. 
        # Using 20 diverse docs here for a good balance.
        data_records = json.load(f)[:20] 
        
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

    # Pre-compute embeddings to save time
    print("  Pre-computing embeddings...")
    for record in data_records:
        record['_emb'] = model.encode(record['sentences'])

    # Sweep parameters
    lambda_values = np.arange(0.0, 2.1, 0.2)
    avg_scores = []

    for lam in tqdm(lambda_values, desc="  Sweeping lambda"):
        chunker = StructureAwareChunker(model, structure_weight=lam, base_threshold=0.6)
        scores = []
        
        for record in data_records:
            # We override the internal split logic to use pre-computed embeddings if possible
            # But for simplicity, we re-instantiate or just call split_text. 
            # Ideally, refactor chunker to accept pre-computed embeddings.
            # Here we just re-run (fast enough for 20 docs).
            chunks = chunker.split_text(record['sentences'])
            
            # Flatten predictions
            pred_labels = []
            for i, c in enumerate(chunks): 
                pred_labels.extend([i] * len(c))
            
            gt = record['gt_labels']
            min_len = min(len(gt), len(pred_labels))
            ami = adjusted_mutual_info_score(gt[:min_len], pred_labels[:min_len])
            scores.append(ami)
        
        avg_scores.append(np.mean(scores))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambda_values, avg_scores, marker='o', linestyle='-', color='#2980b9', linewidth=2)
    
    # Highlight Robust Region
    # Assuming robust if score is within 95% of peak
    peak_score = max(avg_scores)
    ax.axhspan(peak_score * 0.95, peak_score * 1.01, color='#2ecc71', alpha=0.1, label='Robust Region')
    
    ax.set_title(r"Parameter Sensitivity: Structure Weight ($\lambda$)", fontsize=12)
    ax.set_xlabel(r"Structure Penalty Weight ($\lambda$)", fontsize=11)
    ax.set_ylabel("Average AMI Score", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    save_path = os.path.join(IMG_OUTPUT_DIR, 'sensitivity_analysis.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()


def main():
    ensure_dir(IMG_OUTPUT_DIR)
    plot_performance_comparison()
    plot_mechanism_heatmap()
    plot_sensitivity_analysis()
    print("\nAll visualizations generated successfully.")

if __name__ == "__main__":
    main()