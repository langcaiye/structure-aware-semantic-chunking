import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

class StructureAwareChunker:
    def __init__(self, 
                 embedding_model, 
                 base_threshold: float = 0.6, 
                 structure_weight: float = 1.0,
                 c: float = 0.9,
                 init_constant: float = 1.5):
        """
        Args:
            embedding_model: SentenceTransformer model
            base_threshold: 'hard_thr' (default 0.6)
            structure_weight: 0.0 = Original MaxMin; >0.0 = Structure-Aware
            c: Original MaxMin hyperparameter
            init_constant: Original MaxMin hyperparameter
        """
        self.model = embedding_model
        self.fixed_threshold = base_threshold
        self.structure_weight = structure_weight
        self.c = c
        self.init_constant = init_constant

    def _get_structure_score(self, sentence: str) -> float:
        """
        Heuristic to detect structural boundaries.
        Returns a score between 0.0 and 1.0.
        """
        s = sentence.strip()
        
        # Rule 1: Markdown Headers
        if re.match(r'^#+\s', s): return 1.0
        # Rule 2: Numbered Lists (e.g. "1. Introduction")
        if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', s) and len(s) < 100: return 0.8
        # Rule 3: Uppercase Headers
        if s.isupper() and len(s) < 50: return 0.6
        # Rule 4: Short phrases ending in colon
        if s.endswith(':') and len(s) < 30: return 0.5
            
        return 0.0

    def split_text(self, sentences: List[str]) -> List[List[str]]:
        """
        Modified version of the original MaxMin process_sentences function.
        Logic is IDENTICAL to original when structure_weight=0.0.
        """
        if not sentences:
            return []

        # 1. Compute Embeddings
        embeddings = self.model.encode(sentences)
        
        # --- BELOW IS THE ORIGINAL ALGORITHM LOGIC ---
        # We define sigmoid exactly as in the original code
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        paragraphs = []
        current_paragraph = [sentences[0]]
        cluster_start, cluster_end = 0, 1
        pairwise_min = -float('inf')

        for i in range(1, len(sentences)):
            cluster_embeddings = embeddings[cluster_start:cluster_end]

            if cluster_end - cluster_start > 1:
                new_sentence_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]

                # Adjust threshold based on cluster size and similarity
                adjusted_threshold = pairwise_min * self.c * sigmoid((cluster_end - cluster_start) - 1)
                new_sentence_similarity = np.max(new_sentence_similarities)
                
                # Use the minimum of the minimum similarities and the pairwise_min
                pairwise_min = min(np.min(new_sentence_similarities), pairwise_min)
            else:
                adjusted_threshold = 0
                # Use an initial constant when there's only one sentence in the cluster
                pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
                new_sentence_similarity = self.init_constant * pairwise_min

            # --- PATCH START: Structure Penalty ---
            # This is the ONLY logic change.
            # If structure_weight is 0, penalty is 0, and logic remains 100% original.
            penalty = self._get_structure_score(sentences[i]) * self.structure_weight
            final_similarity = new_sentence_similarity - penalty
            # --- PATCH END ---

            # Decide whether to add the sentence to the current paragraph or start a new one
            # (Replaced 'new_sentence_similarity' with 'final_similarity')
            if final_similarity > max(adjusted_threshold, self.fixed_threshold):
                current_paragraph.append(sentences[i])
                cluster_end += 1
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [sentences[i]]
                cluster_start, cluster_end = i, i + 1
                pairwise_min = -float('inf')

        # Append the last paragraph
        paragraphs.append(current_paragraph)
        return paragraphs