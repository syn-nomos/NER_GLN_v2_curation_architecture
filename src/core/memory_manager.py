
import numpy as np
import torch
import streamlit as st
from rapidfuzz import process, fuzz
from typing import List, Dict, Any, Optional
import time

from src.database.db_manager import DBManager
from src.core.vector_memory import VectorMemory
from src.core.augmented_embeddings import AugmentedEmbeddingBuilder

class MemoryManager:
    """
    The Central Intelligence of the Annotation System.
    Handles:
    1. Memory Indexing (Warm-up for CoNLL/New Data)
    2. Hybrid Retrieval (Fuzzy + Vector + Exact)
    3. Feedback Propagation
    """
    
    def __init__(self, db: DBManager, roberta_model=None):
        self.db = db
        self.roberta = roberta_model # The actual loaded model instance
        
        # Components
        self.vector_memory = VectorMemory(db.db_path)
        self.emb_builder = AugmentedEmbeddingBuilder()
        
        # Cache for Fuzzy Search
        self.known_entities_cache = [] # List of unique texts
        self.refresh_string_cache()
        
    def refresh_string_cache(self):
        """Loads all unique accepted entity texts for fast fuzzy matching."""
        try:
            conn = self.db.conn
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT text_span, label FROM annotations WHERE is_accepted = 1 OR is_golden = 1")
            self.known_entities_cache = [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            print(f"Memory Cache Error: {e}")

    def rebuild_all_vectors(self, progress_callback=None):
        """
        Forces a full rebuild of the vector index.
        1. Clears existing vectors.
        2. Re-runs indexing.
        """
        print("🛑 Clearing database vectors...")
        self.db.clear_all_vectors()
        
        print("🔄 Starting full rebuild...")
        return self.index_missing_vectors(progress_callback)

    def index_missing_vectors(self, progress_callback=None):
        """
        The 'Warm-Up' Protocol. 
        Finds annotations without vectors (e.g. from CoNLL import) and generates them.
        """
        if not self.roberta:
            return "Model not loaded"
            
        missing_anns = self.db.get_annotations_without_vectors()
        total = len(missing_anns)
        
        if total == 0:
            return "Memory is up to date."
            
        print(f"🧠 Indexing {total} missing memories...")
        
        count = 0
        for i, ann in enumerate(missing_anns):
            try:
                # 1. Get Embeddings for full sentence
                full_text = ann['full_text']
                
                # Use updated method in RobertaNER that returns tuple
                tokens, offsets, embeddings = self.roberta.get_embeddings_and_offsets(full_text)
                
                # Find token span
                start_char = ann['start_char']
                end_char = ann['end_char']
                
                start_token = -1
                end_token = -1
                
                # Simple alignment logic
                for idx, (os, oe) in enumerate(offsets):
                    if os == start_char:
                        start_token = idx
                    if oe == end_char:
                        end_token = idx + 1 # exclusive
                
                # Fallback if exact match fail
                if start_token == -1:
                    for idx, (os, oe) in enumerate(offsets):
                        if os <= start_char and oe > start_char:
                            start_token = idx
                            break
                if end_token == -1:
                    for idx, (os, oe) in enumerate(offsets):
                        if os < end_char and oe >= end_char:
                            end_token = idx + 1
                            break
                            
                if start_token != -1 and end_token != -1:
                    # Build Vector
                    emb_np = embeddings.numpy()[0] # Batch 0
                    
                    context_vec = self.emb_builder._get_context_vector(emb_np, start_token, end_token)
                    type_vec = self.emb_builder.get_type_vector(ann['label'])
                    
                    final_vec = np.concatenate([context_vec, type_vec])
                    
                    # Save blob
                    vector_blob = final_vec.tobytes()
                    self.db.update_annotation_vector(ann['id'], vector_blob)
                    
                    # CRITICAL: Also update the Entity Centroid (The Shared Memory)
                    self.db.update_entity_centroid(
                        text_span=ann['text_span'], 
                        label=ann['label'], 
                        new_vector_blob=vector_blob
                    )

                    count += 1
                
            except Exception as e:
                print(f"Indexing Error on {ann['id']}: {e}")
                if "index out of range" in str(e):
                    print("💡 HINT: This usually means the Tokenizer has a larger vocabulary than the Model.")
                
            if progress_callback and i % 5 == 0:
                progress_callback(i / total)
        
        # Explicit commit at the end
        if self.db.conn: self.db.conn.commit()

        return f"Successfully indexed {count} / {total} memories."
        if self.db.conn: self.db.conn.commit()
                
        # Reload vector memory after update
        self.vector_memory.load_memory()
        return f"Successfully indexed {count}/{total} memories."

    def find_suggestions(self, text, threshold_fuzzy=85, threshold_vector=0.85):
        """
        Multilayer Search:
        1. Fuzzy: Checks if string is similar to known accepted entities.
        2. Vector: Checks if context is similar (if model available).
        """
        suggestions = []
        
        # --- Layer 1: Fuzzy String Match (Typo Tolerance) ---
        # Get list of unique text strings from cache
        candidates = list(set([x['text_span'] for x in self.known_entities_cache]))
        
        # RapidFuzz extract
        # Returns list of (match, score, index)
        matches = process.extract(text, candidates, limit=5, scorer=fuzz.ratio)
        
        for match_text, score, idx in matches:
            if score >= threshold_fuzzy:
                # Find label for this text (simple lookup)
                # There might be multiple labels for same text (e.g. 'Apple' ORG vs 'Apple' FRUIT)
                # We return all known variants
                relevant = [x for x in self.known_entities_cache if x['text_span'] == match_text]
                for r in relevant:
                    suggestions.append({
                        'text': match_text,
                        'label': r['label'],
                        'confidence': score / 100.0,
                        'reason': f"Fuzzy Match ({score:.1f}%)",
                        'source': 'Memory (Fuzzy)'
                    })

        # --- Layer 2: Vector Match (Semantics) ---
        if self.roberta:
            try:
                # Embed the query text as a standalone phrase
                # We want to compare the "Meaning of the Phrase" against the "Meanings in Memory"
                _, _, emb = self.roberta.get_embeddings_and_offsets(text)
                # emb is [1, seq, 768]. We ignore CLS(0) and SEP(-1) usually, 
                # but for short phrases, taking the mean of everything is fine or just 1:-1.
                
                # tensor -> numpy
                emb_np = emb.detach().cpu().numpy()[0] # shape (seq, 768)
                
                # Exclude CLS/SEP if possible (assuming seq >= 3 for [CLS] word [SEP])
                if emb_np.shape[0] >= 3:
                    query_vec = np.mean(emb_np[1:-1], axis=0)
                else:
                    query_vec = np.mean(emb_np, axis=0) # Fallback
                
                # Search Vector Memory
                vec_matches = self.vector_memory.find_similar(query_vec, k=5, threshold=threshold_vector)
                
                for m in vec_matches:
                    suggestions.append({
                        'text': m['text'],
                        'label': m['label'],
                        'confidence': m['similarity'],
                        'reason': f"Vector Sim ({m['similarity']:.2f})",
                        'source': 'Memory (Vector)'
                    })
            except Exception as e:
                print(f"Vector Search Error: {e}")

        # --- Layer 3: Fusion & De-duplication ---
        # We merge suggestions based on Label + Text overlap
        final_suggestions = []
        seen = set()
        
        for s in suggestions:
            key = (s['text'], s['label'])
            if key not in seen:
                final_suggestions.append(s)
                seen.add(key)
            else:
                # If we have a duplicate (e.g. found by both Fuzzy and Vector), 
                # we could boost confidence or keep the highest one.
                # For simplicity here, we stick with the first one found (Fuzzy usually first)
                pass
                
        return final_suggestions 
        # This is usually done inside the Annotator loop where we have the current sentence embedding.
        
        return suggestions

    def propagate(self, text, label, action):
        """Wrapper for DB propagation."""
        count = self.db.propagate_action(text, label, action)
        return count
