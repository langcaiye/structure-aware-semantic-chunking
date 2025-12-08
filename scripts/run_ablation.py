import sys
import os
import json
import torch
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import adjusted_mutual_info_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core import StructureAwareChunker
from sentence_transformers import SentenceTransformer

# 覆盖 core 中的 score 函数来模拟 Ablation
class AblationChunker(StructureAwareChunker):
    def __init__(self, model, mode='full'):
        super().__init__(model, base_threshold=0.6, structure_weight=1.0)
        self.mode = mode
    
    def _get_structure_score(self, sentence: str) -> float:
        s = sentence.strip()
        is_header = bool(re.match(r'^#+\s', s))
        is_list = bool(re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', s) and len(s) < 100)
        is_upper = bool(s.isupper() and len(s) < 50)
        
        if self.mode == 'no_penalty':
            return 0.0
        elif self.mode == 'headers_only':
            return 1.0 if is_header else 0.0
        elif self.mode == 'headers_lists':
            if is_header: return 1.0
            if is_list: return 0.8
            return 0.0
        else: # full
            return super()._get_structure_score(sentence)

def run_ablation():
    with open("../data/benchmark_50.json", "r") as f:
        data = json.load(f)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    
    modes = ['no_penalty', 'headers_only', 'headers_lists', 'full']
    results = {}
    
    print("Running Ablation Study...")
    for mode in modes:
        chunker = AblationChunker(model, mode=mode)
        scores = []
        for record in tqdm(data, desc=mode):
            chunks = chunker.split_text(record['sentences'])
            pred = []
            for i, c in enumerate(chunks): pred.extend([i]*len(c))
            gt = record['gt_labels']
            min_len = min(len(gt), len(pred))
            scores.append(adjusted_mutual_info_score(gt[:min_len], pred[:min_len]))
        results[mode] = sum(scores) / len(scores)
        
    print("\n=== Table 2 Data ===")
    base = results['no_penalty']
    print(f"No Penalty (Base): {base:.4f}")
    print(f"Headers Only:      {results['headers_only']:.4f} (Delta: +{results['headers_only']-base:.4f})")
    print(f"Headers + Lists:   {results['headers_lists']:.4f} (Delta: +{results['headers_lists']-base:.4f})")
    print(f"Full Method:       {results['full']:.4f} (Delta: +{results['full']-base:.4f})")

if __name__ == "__main__":
    run_ablation()
