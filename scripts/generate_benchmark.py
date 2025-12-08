import wikipediaapi
import re
import json
import os
import random
from tqdm import tqdm

OUTPUT_FILE = "../data/benchmark_50.json"
# TOPICS_BY_DOMAIN
TOPICS_BY_DOMAIN = {
    "Law & Finance": [
        "Contract law", "Tort", "Intellectual property", "Corporate law", "Bankruptcy",
        "Financial statement", "Balance sheet", "Cash flow statement", "Audit", "Tax",
        "General Data Protection Regulation", "Sarbanes–Oxley Act", "Securities and Exchange Commission"
    ],
    "CS & AI": [
        "Deep learning", "Machine learning", "Artificial intelligence", "Transformer (machine learning model)",
        "Natural language processing", "Computer vision", "Reinforcement learning",
        "Operating system", "Linux kernel", "Distributed computing", "Database", "Cloud computing"
    ],
    "Medicine": [
        "Cancer", "Diabetes mellitus", "Immune system", "DNA", "Virus",
        "Cardiology", "Neurology", "Pharmacology", "Vaccine", "Antibiotic"
    ],
    "Physics & Eng": [
        "Quantum mechanics", "General relativity", "Thermodynamics", "Civil engineering", "Electrical engineering",
        "Aerospace engineering", "Nanotechnology", "Renewable energy", "Nuclear power"
    ],
    "History & Soc": [
        "World War II", "Industrial Revolution", "United Nations", "European Union", "Economics", "Inflation"
    ]
}

def build_dataset():
    wiki = wikipediaapi.Wikipedia(
        user_agent='StructureAwareBenchmark/4.0 (fix_uppercase_trap)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    
    data_records = []
    print(f"Generating optimized benchmark with semantic traps...")
    
    for domain, topics in TOPICS_BY_DOMAIN.items():
        for title in tqdm(topics, desc=domain):
            page = wiki.page(title)
            if not page.exists(): continue
                
            text_raw = page.text
            paragraphs = [p for p in text_raw.split('\n') if len(p) > 60]
            paragraphs = paragraphs[:8] 
            
            if len(paragraphs) < 3: continue
            
            doc_sentences = []
            doc_labels = [] 
            
            for i, para in enumerate(paragraphs):
                style = i % 3 
                
                # 1. Header
                if style == 0:
                    # Type A: Markdown (Strong semantic trap)
                    header = f"# Section {i+1}: Detailed Analysis of {title}"
                elif style == 1:
                    # Type B: Numbered List (Strong trap)
                    header = f"{i+1}. {title} Implementation and Core Concepts"
                else:
                    # Type C: Uppercase (CRITICAL FIX)
                    clean_title = title.split('(')[0].strip().upper() 
                    header = f"PART {i+1}: {clean_title} SYSTEM ARCHITECTURE"
                
                doc_sentences.append(header)
                doc_labels.append(i)
                
                # 2. 构造正文
                sents = re.split(r'(?<=[.!?]) +', para)
                for s in sents:
                    if len(s.strip()) > 5:
                        doc_sentences.append(s)
                        doc_labels.append(i)
            
            data_records.append({
                "doc_id": title,
                "domain": domain,
                "sentences": doc_sentences,
                "gt_labels": doc_labels
            })
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data_records, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Diverse Benchmark saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset()