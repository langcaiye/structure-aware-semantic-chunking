import numpy as np
import re
from typing import List, Dict, Any

class StructureAwareChunker:
    def __init__(self, 
                 embedding_model, 
                 base_threshold: float = 0.6, 
                 structure_weight: float = 1.0):
        """
        Args:
            embedding_model: SentenceTransformer model instance (or anything with .encode())
            base_threshold: The cosine similarity threshold for merging (default 0.6)
            structure_weight: The penalty weight for structural boundaries (default 1.0). 
                              Set to 0.0 to behave like the original Max-Min chunker.
        """
        self.model = embedding_model
        self.base_threshold = base_threshold
        self.structure_weight = structure_weight

    def _get_structure_score(self, sentence: str) -> float:
        """Heuristic function to detect structural boundaries."""
        s = sentence.strip()
        
        # Rule 1: Markdown Headers (# Section)
        if re.match(r'^#+\s', s):
            return 1.0
        
        # Rule 2: Numbered Lists / Section Titles (e.g., "1. Introduction")
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', s) and len(s) < 100:
            return 0.8
        
        # Rule 3: Uppercase short headings (e.g., "ABSTRACT")
        if s.isupper() and len(s) < 50:
            return 0.6
            
        # Rule 4: Short phrases ending with colon
        if s.endswith(':') and len(s) < 30:
            return 0.5
            
        return 0.0

    def _cosine_similarity(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def split_text(self, sentences: List[str]) -> List[List[str]]:
        """
        Main function to split sentences into chunks.
        """
        if not sentences:
            return []

        # 1. Compute Embeddings Batch
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_emb = [embeddings[0]]

        for i in range(1, len(sentences)):
            sen = sentences[i]
            emb = embeddings[i]

            # --- Logic: Calculate Similarity ---
            # To be robust, we compare with the LAST sentence of the current chunk
            # (Simplification of Max-Min for efficiency, effectively similar)
            prev_emb = current_chunk_emb[-1]
            sim = self._cosine_similarity(emb, prev_emb)
            
            # --- Logic: Apply Structure Penalty ---
            structure_penalty = self._get_structure_score(sen) * self.structure_weight
            
            # Adjusted Score
            final_score = sim - structure_penalty
            
            # --- Decision ---
            if final_score >= self.base_threshold:
                # Merge
                current_chunk.append(sen)
                current_chunk_emb.append(emb)
            else:
                # Split
                chunks.append(current_chunk)
                current_chunk = [sen]
                current_chunk_emb = [emb]
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

# Wrapper for easy testing
def get_chunker(device='cpu'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    return model
