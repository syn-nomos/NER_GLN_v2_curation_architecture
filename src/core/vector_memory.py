import sqlite3
import numpy as np
import json

class VectorMemory:
    def __init__(self, db_path="data/annotations.db"):
        self.db_path = db_path
        self.vectors = None
        self.metadata = []
        self.prototypes = {} 
        
        self.rejected_vectors = None
        self.rejected_metadata = []
        
        self.load_memory()

    def load_memory(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Φόρτωση Prototypes (Έτοιμα από τη βάση!)
        cursor.execute("SELECT label, vector FROM prototypes")
        for r in cursor.fetchall():
            self.prototypes[r[0]] = np.frombuffer(r[1], dtype=np.float32)
            
        # 2. Φόρτωση Vectors (Accepted / Golden)
        cursor.execute("""
            SELECT vector, label, text_span, frequency, trigger_text
            FROM annotations 
            WHERE vector IS NOT NULL AND (is_accepted=1 OR is_golden=1)
        """)
        rows = cursor.fetchall()
        
        vec_list = []
        meta_list = []
        
        for r in rows:
            vec_blob, label, text, freq, trig = r
            vector = np.frombuffer(vec_blob, dtype=np.float32)
            
            # DIMENSION CHECK (Critical for V2)
            # We expect 768 dimensions. If old vector (2313), skip or pad?
            if vector.shape[0] != 768:
                continue
                
            vec_list.append(vector)
            meta_list.append({
                'label': label, 
                'text_span': text, 
                'frequency': freq,
                'trigger_text': trig
            })
            
        if vec_list:
            self.vectors = np.array(vec_list)
            self.metadata = meta_list
            print(f"🧠 Memory Loaded: {len(self.vectors)} accepted entities, {len(self.prototypes)} prototypes.")
        else:
            print("⚠️ Accepted Memory is empty.")
            
        # 3. Load Rejected Vectors (Negative Memory)
        cursor.execute("""
            SELECT vector, label, text_span 
            FROM annotations 
            WHERE vector IS NOT NULL AND is_rejected=1
        """)
        rows_rej = cursor.fetchall()
        
        rej_vec_list = []
        rej_meta_list = []
        
        for r in rows_rej:
            vec_blob, label, text = r
            vector = np.frombuffer(vec_blob, dtype=np.float32)
            if vector.shape[0] != 768: continue
            rej_vec_list.append(vector)
            rej_meta_list.append({'label': label, 'text': text})
            
        if rej_vec_list:
            self.rejected_vectors = np.array(rej_vec_list)
            self.rejected_metadata = rej_meta_list
            print(f"🛡️ Negative Memory Loaded: {len(self.rejected_vectors)} rejected items.")
        
        conn.close()

    def find_similar(self, query_vector, k=10, threshold=0.0):
        if self.vectors is None: return []

        # ---------------------------------------------------------
        #  SIMPLE V2 SEARCH
        # ---------------------------------------------------------
        
        # 1. Norms
        norm_mem = np.linalg.norm(self.vectors, axis=1)
        norm_q = np.linalg.norm(query_vector)
        
        # 2. Cosine Sim
        denom = norm_mem * norm_q
        denom[denom == 0] = 1e-9
        
        sims = np.dot(self.vectors, query_vector) / denom
        
        # Sort
        indices = np.argsort(sims)[::-1][:k]
        
        results = []
        for idx in indices:
            score = sims[idx]
            if score < threshold: break
            
            meta = self.metadata[idx]
            results.append({
                'label': meta['label'],
                'text': meta['text_span'],
                'similarity': float(score),
                'frequency': meta['frequency'],
                'trigger': meta['trigger_text']
            })
            
        return results
        final_scores = (0.5 * sim_entity) + (0.25 * sim_prev) + (0.25 * sim_next)

        # 6. Retrieve Top-K
        top_k_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = float(final_scores[idx])
            if score >= threshold:
                res = self.metadata[idx].copy()
                res['similarity'] = score
                # Debug info
                res['debug_sim_entity'] = float(sim_entity[idx])
                res['debug_sim_ctx'] = float((sim_prev[idx] + sim_next[idx]) / 2)
                results.append(res)
                
        return results

    def get_prototype_similarity(self, query_vector):
        scores = {}
        if not self.prototypes: return scores
        
        norm_q = np.linalg.norm(query_vector)
        if norm_q == 0: return scores
        
        for label, centroid in self.prototypes.items():
            norm_c = np.linalg.norm(centroid)
            sim = np.dot(query_vector, centroid) / (norm_q * norm_c)
            scores[label] = float(sim)
        return scores

    def check_is_rejected(self, query_vector, label, threshold=0.95):
        """
        Check if the query vector is highly similar to a rejected item with the SAME label.
        Returns (True, matching_text) if likely a recurring false positive.
        """
        if self.rejected_vectors is None: return False, None
        
        norm_q = np.linalg.norm(query_vector)
        norms_r = np.linalg.norm(self.rejected_vectors, axis=1)
        
        denom = norms_r * norm_q
        denom[denom == 0] = 1e-9
        
        sims = np.dot(self.rejected_vectors, query_vector) / denom
        
        # Check only potential matches above threshold
        matches_indices = np.where(sims >= threshold)[0]
        
        for idx in matches_indices:
            rej_label = self.rejected_metadata[idx]['label']
            if rej_label == label:
                return True, self.rejected_metadata[idx]['text']
                
        return False, None