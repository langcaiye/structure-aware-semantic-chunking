# Structure-Aware Semantic Chunking

[![Paper](https://img.shields.io/badge/Paper-Zenodo-green)](https://doi.org/10.5281/zenodo.17797912)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Don't let your RAG system break your tables.**

This repository contains the official implementation of the paper *"Structure-Aware Semantic Chunking: A Hybrid Penalty Mechanism for Document Segmentation"*.

### 💡 The Problem
Traditional semantic chunking methods (like standard Max-Min) are "structure-blind". They merge distinct sections (headers, lists) if the semantic embedding is similar, destroying the logical layout of financial and legal documents.

![Failure Case Demo](images/heatmap_real.png)
*(Note: Please ensure you place your heatmap visualization image in the images/ folder)*

### ✨ Our Solution
We introduce a lightweight **structure penalty term** into the clustering process. It forces segmentation at explicit boundaries (Headers, Markdown, Lists) regardless of semantic similarity.

### 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate the benchmark dataset**
   ```bash
   cd scripts
   python generate_benchmark.py
   ```
3. **Run the evaluation**
This compares the Baseline (Max-Min) against our Structure-Aware method.
   ```bash
   python run_eval.py
   ```
### 📊 Benchmark Results (N=50 Documents)
Evaluated on a diverse corpus of 50 technical documents across Law, Finance, Medicine, and CS.
| Method |	Average AMI Score |
| :---| :---: |
| Baseline (Semantic Only) |	0.5656 |
| Structure-Aware (Ours) |	0.7574 |
| Improvement |	+0.1918 |

### 📂 Dataset
The benchmark dataset constructed for this research is available in data/benchmark_50.json. It contains 50 structured documents with ground-truth section labels.

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
