import wikipediaapi
import re
import json
import os
from tqdm import tqdm

# 1. Configuration (使用字典结构来明确领域)
OUTPUT_FILE = "../data/benchmark_50.json"
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
        user_agent='StructureAwareBenchmark/2.0 (opensource@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    
    data_records = []
    total_articles = sum(len(v) for v in TOPICS_BY_DOMAIN.values())
    print(f"Downloading {total_articles} articles across {len(TOPICS_BY_DOMAIN)} domains...")
    
    # 遍历字典
    for domain, topics in TOPICS_BY_DOMAIN.items():
        print(f"Processing Domain: {domain}...")
        for title in tqdm(topics):
            page = wiki.page(title)
            if not page.exists():
                continue
                
            text_raw = page.text
            paragraphs = [p for p in text_raw.split('\n') if len(p) > 60]
            paragraphs = paragraphs[:8] 
            
            if len(paragraphs) < 3: continue
            
            doc_sentences = []
            doc_labels = [] 
            
            for i, para in enumerate(paragraphs):
                # Inject Header
                header = f"# Section {i+1}: Analysis of {title} part {i}"
                doc_sentences.append(header)
                doc_labels.append(i)
                
                # Split sentences
                sents = re.split(r'(?<=[.!?]) +', para)
                for s in sents:
                    if len(s.strip()) > 5:
                        doc_sentences.append(s)
                        doc_labels.append(i)
            
            data_records.append({
                "doc_id": title,
                "domain": domain,  # <--- 新增：保存领域标签
                "sentences": doc_sentences,
                "gt_labels": doc_labels
            })
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data_records, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Benchmark dataset saved to {OUTPUT_FILE} with {len(data_records)} documents.")

if __name__ == "__main__":
    build_dataset()