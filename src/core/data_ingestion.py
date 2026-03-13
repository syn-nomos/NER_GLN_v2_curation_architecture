import spacy
import json
import PyPDF2
from typing import List, Dict, Optional
import io

class DataIngestion:
    """
    Handles file parsing (Text, PDF, CoNLL) and ingestion into the DB.
    Uses Spacy for robust sentence splitting in raw texts.
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.nlp = None

    def _load_spacy(self):
        """Lazy loader for Spacy model with custom Greek abbreviation rules."""
        if self.nlp is None:
            try:
                # Prefer Greek model
                self.nlp = spacy.load("el_core_news_sm")
            except Exception:
                print("Warning: 'el_core_news_sm' not found. Falling back to blank model.")
                self.nlp = spacy.blank("el")
                self.nlp.add_pipe("sentencizer")

            # --- CUSTOM TOKENIZER EXCEPTIONS FOR GREEK LEGAL TEXT ---
            # prevents splitting on: Αριθ., Τ.Κ., κ.λπ., κ.α., Π.Δ., Ν.
            from spacy.symbols import ORTH
            
            abbreviations = [
                "Αριθ.", "αριθ.", "αρ.", "Αρ.", "Τ.Κ.", "τ.κ.",
                "κ.κ.", "κ.", "Κ.", "Π.Δ.", "π.δ.", "Ν.", "ν.", 
                "ΦΕΚ", "τ.μ.", "χλμ.", "εκ.", "χιλ.", 
                "Α.Π.", "Ε.Ε.", "Ε.Κ.", "παρ.", "Παρ.",
                "τηλ.", "φαξ", "Σελ.", "σελ.", "εικ.", "Δρ."
            ]
            
            # If we are using the parser/sentencizer, we must ensure these tokens 
            # are NOT seen as end-of-sentence.
            # In Spacy 3.x, modifying tokenizer.rules works for tokenization,
            # but sentence segmentation (parser/sentencizer) relies on punctuation.
            
            # We add them as special case exceptions to the tokenizer.
            for abbr in abbreviations:
                self.nlp.tokenizer.add_special_case(abbr, [{ORTH: abbr}])


    def process_file_upload(self, file_obj, source_type: str) -> int:
        """Main entry point for UI uploads."""
        filename = file_obj.name
        
        if source_type == "PDF Document (.pdf)":
            return self._process_pdf(file_obj, filename)
        elif source_type == "Raw Text (.txt)":
            text = file_obj.getvalue().decode("utf-8")
            return self._process_raw_text(text, filename, "TXT")
        elif source_type == "CONLL": # Internal call mostly
            text = file_obj.getvalue().decode("utf-8")
            return self.process_conll(text, filename)
        return 0

    def _process_pdf(self, file_stream, filename: str) -> int:
        reader = PyPDF2.PdfReader(file_stream)
        full_text = ""
        for page in reader.pages:
            extract = page.extract_text()
            if extract:
                full_text += extract + "\n"
        
        return self._process_raw_text(full_text, filename, file_type="PDF")

    def _process_raw_text(self, text: str, filename: str, file_type: str = "TXT") -> int:
        self._load_spacy()
        
        # 1. Use filename as DocID
        doc_id = filename
        
        # 2. Split Sentences using Spacy
        doc = self.nlp(text)
        
        count = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            # Basic validation
            if not sent_text or len(sent_text) < 5:
                continue
            
            # Store tokens as JSON for consistency
            tokens = [token.text for token in sent]
            
            self.db.add_sentence(
                text=sent_text,
                source_doc=doc_id,
                tokens=tokens,
                split="pending", # Default for raw imports
                meta={"char_len": len(sent_text)}
            )
            count += 1
            
        return count

    def process_conll(self, content: str, filename: str) -> int:
        """
        Parses CoNLL format (Word + Tag), reconstructs sentences, and saves annotations.
        """
        # 1. Use filename as DocID (No separate document table)
        doc_id = filename
        
        # 2. Determine Split from filename
        lower_name = filename.lower()
        if "test" in lower_name: split = "test"
        elif "dev" in lower_name or "valid" in lower_name: split = "dev"
        elif "train" in lower_name: split = "train"
        else: split = "pending"
        
        # Robust Splitting: Line-by-Line approach
        # This handles \n, \r\n, and lines with just whitespace correctly
        lines = content.splitlines()
        
        current_tokens = []
        current_tags = []
        count = 0
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # End of sentence detected
                if current_tokens:
                    self._save_conll_sentence(doc_id, split, current_tokens, current_tags)
                    count += 1
                    current_tokens = []
                    current_tags = []
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_tags.append(parts[-1])
        
        # Save last sentence if file doesn't end with newline
        if current_tokens:
             self._save_conll_sentence(doc_id, split, current_tokens, current_tags)
             count += 1
        
        return count

    def _save_conll_sentence(self, doc_id, split, tokens, tags):
        """Helper to save a parsed sentence to DB."""
        text = " ".join(tokens)
        
        # Save Sentence
        # We pass document_id as source_doc
        sent_id = self.db.add_sentence(
            text=text,
            source_doc=doc_id,
            tokens=tokens,
            split=split
        )
        
        # Convert BIO Tags to DB Annotations
        annotations = self._bio_to_annotations(tokens, tags, text)
        
        if annotations:
            self.db.save_annotations(sent_id, annotations, mark_complete=True)

    def _bio_to_annotations(self, tokens: List[str], tags: List[str], text: str) -> List[Dict]:
        """Helper to convert BIO tags to DB annotation objects with character offsets."""
        formatted_anns = []
        
        # 1. Map tokens to character offsets in the reconstructed text
        token_spans = [] # List of (char_start, char_end)
        current_char = 0
        for token in tokens:
            start = current_char
            end = start + len(token)
            token_spans.append((start, end))
            current_char = end + 1 # +1 for the space we added in join()

        # 2. Parse BIO entities
        current_entity = None
        temp_entities = [] # List of {label, start_tok, end_tok}
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_entity:
                    temp_entities.append(current_entity)
                
                label = tag[2:]
                current_entity = {'label': label, 'start_tok': i, 'end_tok': i}
                
            elif tag.startswith('I-') and current_entity:
                # Check for label consistency
                label = tag[2:]
                if label == current_entity['label']:
                    current_entity['end_tok'] = i
                else:
                    # Label mismatch in I- tag (shouldn't happen in valid BIO, but just in case)
                    temp_entities.append(current_entity)
                    current_entity = {'label': label, 'start_tok': i, 'end_tok': i}
            
            elif tag == 'O':
                if current_entity:
                    temp_entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            temp_entities.append(current_entity)
            
        # 3. Convert token indices to character offsets
        for ent in temp_entities:
            s_tok = ent['start_tok']
            e_tok = ent['end_tok']
            
            char_start = token_spans[s_tok][0]
            char_end = token_spans[e_tok][1]
            
            entity_text = text[char_start:char_end]
            
            formatted_anns.append({
                'label': ent['label'],
                'start_offset': char_start,
                'end_offset': char_end,
                'token_start': s_tok,
                'token_end': e_tok,
                'text': entity_text,
                'source': 'imported_conll',
                'is_correct': 1
            })
            
        return formatted_anns
