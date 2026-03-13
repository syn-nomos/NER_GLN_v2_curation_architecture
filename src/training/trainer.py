import os
import torch
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    AutoConfig
)
import evaluate

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActiveTrainer:
    def __init__(self, base_model_path: str, output_dir: str, labels_list: List[str] = None):
        """
        Args:
            base_model_path: Path to the starting model.
            output_dir: Root directory for versions.
            labels_list: Optional list of NEW labels found in data. 
                         The trainer will primarily try to inherit from base model config.
        """
        self.base_model = base_model_path
        self.output_dir = output_dir
        
        # 1. Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_prefix_space=True)
        except:
             self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True)

        # 2. Smart Label Resolution (The "Old App" Logic)
        self.id2label = {}
        self.label2id = {}
        
        try:
            config = AutoConfig.from_pretrained(base_model_path)
            if hasattr(config, "id2label") and config.id2label:
                logger.info(f"♻️ Inheriting {len(config.id2label)} labels from Base Model config.")
                self.id2label = {int(k): v for k, v in config.id2label.items()}
                self.label2id = {v: int(k) for k, v in config.id2label.items()}
            else:
                logger.warning("⚠️ No id2label in config. Starting fresh.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load config: {e}")

        # 3. Merge with provided labels_list if any (Handling new labels)
        if labels_list:
            if not self.label2id:
                # Fresh start
                # Ensure 'O' is 0
                unique_labels = sorted(list(set(labels_list)))
                if 'O' in unique_labels:
                    unique_labels.remove('O')
                final_list = ['O'] + unique_labels
                
                self.id2label = {i: l for i, l in enumerate(final_list)}
                self.label2id = {l: i for i, l in enumerate(final_list)}
            else:
                # Extension mode
                current_max = max(self.id2label.keys())
                existing = set(self.label2id.keys())
                for l in labels_list:
                    if l not in existing:
                        current_max += 1
                        self.id2label[current_max] = l
                        self.label2id[l] = current_max
                        logger.info(f"🆕 Added new label: {l} (ID: {current_max})")

        self.label_list = [self.id2label[i] for i in sorted(self.id2label.keys())]
        
        # Metric calculation
        try:
            self.metric = evaluate.load("seqeval")
        except:
            # Fallback if internet access is restricted, though seqeval usually needs it or local install
            self.metric = None 

    def prepare_data(self, sentences: List[Dict]):
        """
        Converts DB sentence objects (text + offsets) into Tokenized HF Dataset.
        """
        data = {
            "id": [],
            "tokens": [],
            "ner_tags": []
        }
        
        for sent in sentences:
            text = sent['text']
            
            # Simple whitespace tokenization for aligning (heuristic)
            # NOTE: Ideally we use spacy tokenization if available to match DB logic better,
            # but for now we do a whitespace split and align char offsets.
            raw_tokens = text.split()
            
            # Create token offsets
            token_spans = []
            cursor = 0
            for t in raw_tokens:
                try:
                    start = text.index(t, cursor)
                except ValueError:
                    # Fallback
                    start = cursor
                end = start + len(t)
                token_spans.append((start, end))
                cursor = end
            
            # Map Annotations to Tags
            # Initialize all as 'O' (ID 0)
            tags = [0] * len(raw_tokens)
            
            annotations = sent.get('annotations', [])
            for ann in annotations:
                # Filter for trusted only
                if ann.get('source') != 'manual': 
                    continue
                    
                label_name = ann['label']
                # Determine B- tag ID and I- tag ID
                # This assumes labels_list has B-X and I-X logic or we map generic X to B-X
                # Let's handle standard IO B- I- format
                
                b_tag = f"B-{label_name}"
                i_tag = f"I-{label_name}"
                
                # Check if these exist in our label map, else fallback to O or skip
                if b_tag not in self.label2id:
                    # Try to see if plain label exists (maybe not BIO scheme?)
                    if label_name in self.label2id:
                        b_tag_id = self.label2id[label_name]
                        i_tag_id = b_tag_id
                    else:
                        continue 
                else:
                    b_tag_id = self.label2id[b_tag]
                    i_tag_id = self.label2id[i_tag]

                start_char = ann['start_offset']
                end_char = ann['end_offset']
                
                for idx, (t_start, t_end) in enumerate(token_spans):
                    # Check overlap
                    if t_end > start_char and t_start < end_char:
                        if t_start == start_char: # Matches start exactly -> B
                             tags[idx] = b_tag_id
                        elif tags[idx] == 0: # Only overwrite O (don't overwrite existing B)
                             # Logic: if it's the first token we encounter for this entity -> B
                             # This is tricky with whitespace approximation.
                             # Simple heuristic: If it's the first token overlapping -> B, else I
                             if t_start <= start_char: 
                                 tags[idx] = b_tag_id
                             else:
                                 tags[idx] = i_tag_id
            
            data["id"].append(sent['id'])
            data["tokens"].append(raw_tokens)
            data["ner_tags"].append(tags)

        return Dataset.from_dict(data)

    def token_alignment_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            padding="max_length", 
            max_length=128 # Keep regular for specific legal texts
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100) # Ignored
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100) # Ignore sub-tokens or use I-tag
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        if not self.metric: return {}
        
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, train_sentences, val_sentences=None, epochs=3, batch_size=4, learning_rate=2e-5):
        
        # 1. Prepare Datasets
        train_ds = self.prepare_data(train_sentences)
        tokenized_train = train_ds.map(self.token_alignment_function, batched=True)
        
        tokenized_val = None
        if val_sentences:
            val_ds = self.prepare_data(val_sentences)
            tokenized_val = val_ds.map(self.token_alignment_function, batched=True)

        # 2. Setup Model
        model = AutoModelForTokenClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

        # 3. Training Args
        version_name = f"model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path = os.path.join(self.output_dir, version_name)
        
        logging_steps = max(1, len(tokenized_train) // batch_size)
        
        args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "checkpoints"),
            evaluation_strategy="epoch" if tokenized_val else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=logging_steps,
            save_total_limit=1, # Save space
            report_to="none" # No wandb blocking
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # 4. Train
        logger.info(f"Starting training for {epochs} epochs...")
        train_result = trainer.train()
        
        # 5. Save Final Model
        logger.info(f"Saving model to {save_path}")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        return save_path, train_result.metrics
