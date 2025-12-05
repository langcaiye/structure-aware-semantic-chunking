import json
import sys
import os
import torch
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import StructureAwareChunker
from sentence_transformers import SentenceTransformer

DATA_FILE = "../data/benchmark_50.json"
RESULT_FILE = "../data/eval_summary.csv"

def run_evaluation():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run generate_benchmark.py first.")
        return

    with open(DATA_FILE, "r") as f:
        data_records = json.load(f)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading Embedding Model on {device}...")
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    
    chunker_base = StructureAwareChunker(model, base_threshold=0.6, structure_weight=0.0)
    chunker_ours = StructureAwareChunker(model, base_threshold=0.6, structure_weight=1.0)
    
    results = []
    print(f"Evaluating {len(data_records)} documents...")
    
    for record in tqdm(data_records):
        sentences = record['sentences']
        gt = record['gt_labels']
        domain = record.get('domain', 'General')
        
        # Baseline
        chunks_base = chunker_base.split_text(sentences)
        pred_base = []
        for i, c in enumerate(chunks_base): pred_base.extend([i]*len(c))
        
        # Ours
        chunks_ours = chunker_ours.split_text(sentences)
        pred_ours = []
        for i, c in enumerate(chunks_ours): pred_ours.extend([i]*len(c))
        
        # Metrics
        min_len = min(len(gt), len(pred_base), len(pred_ours))
        score_base = adjusted_mutual_info_score(gt[:min_len], pred_base[:min_len])
        score_ours = adjusted_mutual_info_score(gt[:min_len], pred_ours[:min_len])
        
        results.append({
            "Domain": domain,
            "Doc": record['doc_id'],
            "Base AMI": score_base,
            "Ours AMI": score_ours,
            "Delta": score_ours - score_base
        })
        
    df = pd.DataFrame(results)
    
    # --- Aggregation & Saving ---
    # Group by Domain and calculate mean
    domain_summary = df.groupby("Domain")[["Base AMI", "Ours AMI", "Delta"]].mean().reset_index()

    # Save to CSV for visualization
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    domain_summary.to_csv(RESULT_FILE, index=False)
    print(f"\n✅ Evaluation results saved to {RESULT_FILE}")
    
    # --- Print LaTeX Table (Professional Way) ---
    print("\n" + "="*60)
    print("📝 LATEX TABLE GENERATION")
    print("Paste the following code into your LaTeX document:")
    print("="*60 + "\n")
    
    # 重命名列以符合论文的格式
    latex_df = domain_summary.rename(columns={
        "Base AMI": "Baseline (AMI)",
        "Ours AMI": "Ours (AMI)",
        "Delta": "Improvement"
    })
    
    # 格式化数字：保留4位小数，正数加号
    formatters = {
        "Baseline (AMI)": "{:.4f}".format,
        "Ours (AMI)": "{:.4f}".format,
        "Improvement": lambda x: f"+{x:.4f}" if x > 0 else f"{x:.4f}"
    }
    
    # 使用 Pandas 生成 LaTeX 代码 (去除行号 index=False)
    latex_code = latex_df.to_latex(
        index=False, 
        formatters=formatters,
        column_format="|l|c|c|c|", # 对应 LaTeX 的列对齐
        header=True,
        escape=False # 防止特殊字符被转义
    )
    
    # 稍微修饰一下 Pandas 生成的默认代码，加上横线
    latex_code = latex_code.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    
    print(latex_code)
    
    # 打印总平均分
    print("% Overall Average Row:")
    avg_base = df['Base AMI'].mean()
    avg_ours = df['Ours AMI'].mean()
    avg_delta = df['Delta'].mean()
    print(f"\\textbf{{Average}} & \\textbf{{{avg_base:.4f}}} & \\textbf{{{avg_ours:.4f}}} & \\textbf{{+{avg_delta:.4f}}} \\\\ \\hline")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    run_evaluation()