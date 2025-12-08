# Structure-Aware Semantic Chunking

[![Paper](https://img.shields.io/badge/Paper-Zenodo-green)](https://doi.org/10.5281/zenodo.17797912)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Don't let your RAG system break your tables.**

This repository contains the official implementation of the paper *"Structure-Aware Semantic Chunking: A Hybrid Penalty Mechanism for Document Segmentation"*.

### 💡 The Problem
Traditional semantic chunking methods (like standard Max-Min) are "structure-blind". They merge distinct sections (headers, lists) if the semantic embedding is similar, destroying the logical layout of financial and legal documents (e.g., merging a "Implementation" header into a "Coding" paragraph).

### ✨ Our Solution
We introduce a lightweight **structure penalty term** into the clustering process. It forces segmentation at explicit boundaries (Headers, Markdown, Lists) regardless of semantic similarity.

### 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate the benchmark dataset**
   This script downloads technical articles from Wikipedia and injects structural traps (headers, lists) to create a "failure case" benchmark.
   ```bash
   cd scripts
   python generate_benchmark.py
   ```
   *Output: data/benchmark_50.json*
3. **Run the main evaluation**
   This script runs the comparison between the Baseline (Max-Min) and our Structure-Aware method on the 50-document corpus.
   ```bash
   python run_eval.py
   ```
   *Output: data/eval_summary.csv and LaTeX table code.*

4. **Run ablation studies**
   Verify the individual contribution of each rule (Headers vs. Lists vs. Full Method).
   ```bash
   python run_ablation.py
   ```
   *Output: Prints the incremental improvement of each structural rule.*
5. **Visualize results**
   Generate the similarity heatmaps and performance comparison charts used in the paper.
   ```bash
   python visualize_results.py
   ```
   Output: Images saved to ../images/

### 📊 Benchmark Results (N=50 Documents)
We evaluated the method on a diverse corpus of 50 technical documents across Law, Finance, Medicine, and CS. The results demonstrate a significant improvement in recovering ground-truth structure.
| Method | Average AMI Score | Improvement |
| :--- | :---: | :---: |
| Baseline (Semantic Only) | 0.2342 | - |
| **Structure-Aware (Ours)** | **0.9077** | **+0.6735** |

### 📂 Dataset
The benchmark dataset constructed for this research is available in data/benchmark_50.json. It contains 50 structured documents with ground-truth section labels.

### 📂 Project Structure

```text
Structure-Aware-RAG/
├── data/                  # Generated benchmark datasets & evaluation logs
│   ├── benchmark_50.json
│   └── eval_summary.csv
├── images/                # Visualization outputs (heatmaps, plots)
├── scripts/               # Experiment automation scripts
│   ├── generate_benchmark.py  # Step 1: Data generation
│   ├── run_eval.py            # Step 2: Main evaluation (Baseline vs Ours)
│   ├── run_ablation.py        # Step 3: Ablation study
│   └── visualize_results.py   # Step 4: Plotting
├── src/                   # Core algorithm package
│   ├── __init__.py
│   └── core.py            # StructureAwareChunker class implementation
├── requirements.txt       # Python dependencies
└── README.md
```

### 🔗 Citation
If you use this code or dataset, please cite our paper:
   ```bibtex

   @article{ye2025structure,
      title={Structure-Aware Semantic Chunking: A Hybrid Penalty Mechanism for Document Segmentation},
      author={Ruidian Ye},
      year={2025},
      publisher={Zenodo},
      doi={10.5281/zenodo.17797912}
   } 
   ```
### 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
