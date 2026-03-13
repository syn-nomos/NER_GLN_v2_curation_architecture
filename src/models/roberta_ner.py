import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.core.span_utils import group_tokens_to_spans

class RobertaNER:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading RoBERTa model from {model_path} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        except Exception as e:
            print(f"⚠️ Could not load tokenizer from {model_path}. Falling back to 'xlm-roberta-base'. Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True)

        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_path, 
                output_hidden_states=True,
                output_attentions=True
            )
        except Exception as e:
             # Retrigger the exception to be caught by the caller to switch to fallback
             raise e

        self.model.to(self.device)
        self.model.eval()

        # [AUTO-FIX] Validate Vocabulary Compatibility to prevent "Index out of range"
        if hasattr(self.model.config, 'vocab_size'):
            model_vocab = self.model.config.vocab_size
            tokenizer_vocab = self.tokenizer.vocab_size
            
            if tokenizer_vocab > model_vocab + 5000:
                print(f"⚠️ DETECTED VOCAB MISMATCH: Tokenizer ({tokenizer_vocab}) > Model ({model_vocab})")
                print("⚠️ This will cause 'Index out of range' errors. Switching to 'roberta-base' tokenizer...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
                    print(f"✅ Auto-corrected tokenizer to 'roberta-base' (Vocab: {self.tokenizer.vocab_size})")
                except Exception as e:
                    print(f"❌ Failed to auto-correct tokenizer: {e}")

        # Load labels from model config if available
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            self.id2label = self.model.config.id2label
            print(f"✅ Loaded {len(self.id2label)} labels from config.")
        else:
            print("⚠️ No labels found in config. Using default fallback.")
            self.id2label = {
                0: 'O', 1: 'B-ORG', 2: 'I-ORG', 3: 'B-GPE', 4: 'O', 
                5: 'B-LEG-REFS', 6: 'I-LEG-REFS', 7: 'B-LEG-REFS', 8: 'O', 
                9: 'B-PERSON', 10: 'I-PERSON'
            }

    def predict(self, text):
        """Κλασική πρόβλεψη NER"""
        # ... (ο κώδικας παραμένει ίδιος για το predict, αλλά για συντομία 
        # ας εστιάσουμε στη νέα μέθοδο που χρειαζόμαστε)
        # Μπορείς να κρατήσεις την παλιά predict ή να χρησιμοποιήσεις αυτήν την πιο καθαρή ροή:
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.hidden_states[-1][0].cpu().numpy() # (seq_len, 768)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        probs = torch.softmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Εξαγωγή Entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        entities = []
        
        current_entity = None
        
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            label = self.id2label.get(pred, 'O')
            score = prob[pred]
            
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
                
            # B-TAG or I-TAG
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                start_char, end_char = offset_mapping[i]
                # Skip special tokens
                if start_char == 0 and end_char == 0: continue
                
                current_entity = {
                    'label': label[2:],
                    'start': int(start_char),
                    'end': int(end_char),
                    'confidence': float(score),
                    'tokens': [i],
                    'source': 'RoBERTa'
                }
            elif label.startswith('I-') and current_entity:
                # Check if consistent (I-ORG after B-ORG)
                if label[2:] == current_entity['label']:
                    start_char, end_char = offset_mapping[i]
                    # Check if this token is adjacent or overlapping (subword merge)
                    # If offsets are contiguous, merge.
                    current_entity['end'] = max(current_entity['end'], int(end_char))
                    current_entity['confidence'] = min(current_entity['confidence'], float(score))
                    current_entity['tokens'].append(i)
        
        if current_entity:
            entities.append(current_entity)
            
        # Clean up and Filter
        final_entities = []
        
        # [PATCH] Use shared Snapping Logic & Merging
        from src.core.span_utils import snap_entities_to_words, merge_overlapping_spans
        
        # 1. Snap boundaries to full words
        entities = snap_entities_to_words(entities, text)
        
        # 2. Merge overlapping duplicated entities (e.g. fragmented words snapped to same word)
        entities = merge_overlapping_spans(entities, text)
        
        for ent in entities:
            # Extract text (Refreshed after snap/merge)
            span_text = ent.get('text', '')
            
            # 1. Filter empty or whitespace-only
            if not span_text or not span_text.strip():
                continue
                
            # 2. Filter low confidence (noise)
            if ent.get('confidence', 0) < 0.40: 
                continue
                
            # 3. Filter very short trash (e.g. single char 'K' unless special)
            if len(span_text.strip()) < 2:
                # Keep only if it looks like a section symbol or number
                if not span_text.strip().isdigit() and span_text.strip() not in ['§', '%']:
                     continue

            final_entities.append(ent)
            
        return final_entities

    def get_embeddings_and_offsets(self, text):
        """
        Returns (tokens, offsets, embeddings_tensor)
        Useful for building memory vectors.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1].cpu() # [1, seq, 768]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return tokens, offset_mapping, last_hidden_state

    def get_entity_vector(self, text, start_char, end_char):
        """
        Computes the embedding vector for a specific span in the text.
        Returns a numpy array (768,).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use last hidden state
        embeddings = outputs.hidden_states[-1][0].cpu().numpy() # (seq_len, 768)
        
        # Find tokens that fall within start_char and end_char
        target_indices = []
        for i, (s, e) in enumerate(offset_mapping):
            if s == 0 and e == 0: continue # Special tokens like <s>
            
            # Check overlap
            # Token span [s, e]
            # Target span [start_char, end_char]
            if max(s, start_char) < min(e, end_char):
                target_indices.append(i)
                
        if not target_indices:
            # Fallback: global sentence embedding (CLS token)
            return embeddings[0]
            
        # Average the embeddings of the target tokens
        selected_vecs = embeddings[target_indices]
        avg_vec = np.mean(selected_vecs, axis=0)
        
        return avg_vec

    def enrich_spans_with_vectors(self, text, spans):
        """
        Παίρνει έτοιμα spans (π.χ. από Regex) και υπολογίζει τα vectors τους
        τρέχοντας το κείμενο μέσα από το RoBERTa.
        Υποστηρίζει μεγάλα κείμενα μέσω sliding window.
        """
        if not spans:
            return spans

        # 1. Tokenization με Sliding Window
        # Χρησιμοποιούμε stride για να καλύψουμε οντότητες στα όρια
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512, 
            stride=128,
            return_overflowing_tokens=True, 
            return_offsets_mapping=True
        )
        
        # Το inputs μπορεί να έχει πολλά "samples" λόγω του overflow
        # inputs['input_ids'] shape: (num_windows, seq_len)
        
        offset_mappings = inputs.pop("offset_mapping").cpu().numpy()
        # Το overflow_to_sample_mapping δεν το χρειαζόμαστε αν έχουμε μόνο 1 κείμενο εισόδου
        if "overflow_to_sample_mapping" in inputs:
            inputs.pop("overflow_to_sample_mapping")
            
        # 2. Forward Pass (σε batches αν χρειαστεί, αλλά εδώ τα windows είναι το batch)
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            # Catch CUDA errors or any other runtime error
            print(f"⚠️ Error during RoBERTa inference: {e}")
            print(f"⚠️ Switching to CPU permanently to recover.")
            self.device = 'cpu'
            try:
                # RELOAD MODEL ON CPU because GPU context is dead
                print(f"🔄 Reloading model on CPU...")
                from transformers import AutoModelForTokenClassification
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path, 
                    output_hidden_states=True
                ).to('cpu')
                
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except Exception as e2:
                print(f"❌ CPU Fallback failed: {e2}")
                return spans
        
        # (num_windows, seq_len, 768)
        all_token_embeddings = outputs.hidden_states[-1].cpu().numpy()

        # 3. Αντιστοίχιση Spans σε Tokens
        for span in spans:
            char_start = span['start']
            char_end = span['end']
            
            span_vector_found = False
            
            # Ψάχνουμε σε ποιο window ανήκει καλύτερα το span
            for window_idx, offsets in enumerate(offset_mappings):
                token_embeddings = all_token_embeddings[window_idx]
                
                # Βρες τους δείκτες των tokens που καλύπτουν αυτό το span σε αυτό το window
                span_token_vectors = []
                
                # Ελέγχουμε αν το span είναι εντός των ορίων του window (βάσει offsets)
                # Τα offsets έχουν shape (seq_len, 2)
                valid_tokens_mask = (offsets[:, 0] != 0) | (offsets[:, 1] != 0)
                if not np.any(valid_tokens_mask): continue
                
                window_start_char = offsets[valid_tokens_mask][0][0]
                window_end_char = offsets[valid_tokens_mask][-1][1]
                
                # Αν το span είναι τελείως έξω από το window, continue
                if char_end <= window_start_char or char_start >= window_end_char:
                    continue

                for i, (off_start, off_end) in enumerate(offsets):
                    # Skip special tokens
                    if off_start == 0 and off_end == 0: continue
                    
                    # Έλεγχος επικάλυψης
                    if off_end > char_start and off_start < char_end:
                        span_token_vectors.append(token_embeddings[i])
                
                # Αν βρήκαμε tokens
                if span_token_vectors:
                    avg_vector = np.mean(span_token_vectors, axis=0)
                    norm = np.linalg.norm(avg_vector)
                    if norm > 0:
                        avg_vector = avg_vector / norm
                    span['vector'] = avg_vector.tolist()
                    span_vector_found = True
                    break # Βρήκαμε το span σε ένα window, τέλος.
            
            if not span_vector_found:
                # Αν δεν βρέθηκε (π.χ. πολύ μεγάλο span ή weird split), το αγνοούμε
                pass

        return spans