import streamlit as st
import sqlite3
import os
import sys
import pandas as pd
import spacy
from datetime import datetime

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(ROOT_DIR)

from src.database.db_manager import DBManager

st.set_page_config(page_title="Data Export", page_icon="💾")

# Initialize Session State
if "db" not in st.session_state:
    st.warning("Please load a database first.")
    st.stop()

db = st.session_state.db

# Load Spacy
@st.cache_resource
def get_nlp(model_name="el_core_news_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        # Try to download it if missing
        import subprocess
        st.warning(f"Downloading Spacy model '{model_name}'... This creates a delay once.")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
        try:
            return spacy.load(model_name)
        except:
             st.error(f"Failed to load '{model_name}'. Please restart app.")
             return None

nlp = get_nlp()
if not nlp:
    st.stop()

st.title("💾 Export Data")
st.markdown("Export your annotated data to CoNLL format for training.")

# --- Export Settings ---
with st.form("export_form"):
    st.subheader("Configuration")
    
    c1, c2 = st.columns(2)
    output_name = c1.text_input("Output Filename", value=f"export_ver_{datetime.now().strftime('%Y%m%d')}.conll")
    output_dir = c2.text_input("Output Directory", value=os.path.join(ROOT_DIR, "dataset"))
    
    st.markdown("---")
    st.subheader("Filters")
    
    # Splits
    all_splits = ["train", "dev", "test", "pending"]
    selected_splits = st.multiselect("Select Splits to Export", all_splits, default=["train"])
    
    # Source Filter
    export_mode = st.radio("Export Strategy", [
        "Verified Only (Strict) - Only manual/verified annotations",
        "Hybrid (Standard) - Manual + Imported + Auto (All active annotations)",
        "Review Mode - Only export sentences marked as 'annotated'"
    ])
    
    submitted = st.form_submit_button("🚀 Generate Export")

if submitted:
    full_path = os.path.join(output_dir, output_name)
    
    # 1. Fetch Sentences
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Filter by split
    # IGNORE SPLITS - EXPORT EVERYTHING IN DB
    cursor.execute("SELECT id, text, dataset_split, status FROM sentences")
    rows = cursor.fetchall()
    
    if not rows:
        st.error("No sentences found matching criteria.")
    else:
        st.info(f"Processing {len(rows)} sentences...")
        
        # 2. Process & Write
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                count_exported = 0
                
                # Progress bar
                prog_bar = st.progress(0)
                
                for i, (sid, text, split, status) in enumerate(rows):
                    # Get Annotations (ADJUSTED COLUMN NAMES)
                    # DB uses: label, start_char, end_char, text_span, source_agent
                    cursor.execute("SELECT label, start_char, end_char, text_span, source_agent, is_accepted FROM annotations WHERE sentence_id = ? AND (is_rejected = 0 OR is_rejected IS NULL)", (sid,))
                    anns = []
                    for r in cursor.fetchall():
                        label = r[0].replace('_', '-') if r[0] else r[0]
                        anns.append({
                            "label": label,
                            "start": r[1],
                            "end": r[2],
                            "text": r[3],
                            "source_agent": r[4],
                            "is_accepted": r[5]
                        })
                    
                    # Filter Annotations based on Mode
                    valid_anns = []
                    for a in anns:
                        # Ensure keys are correct
                        if "Verified Only" in export_mode:
                            if a.get('is_accepted') == 1: # Strict check? Source?
                                valid_anns.append(a)
                        else:
                            # Hybrid: Allow all current annotations in DB
                            valid_anns.append(a)
                            
                    # OVERLAP RESOLUTION STRATEGY
                    # Policy:
                    # 1. Accepted/Manual annotations take precedence.
                    # 2. Then Length (Longest match).
                    valid_anns.sort(key=lambda x: (
                        x.get('is_accepted', 0) or (1 if x.get('source_agent') == 'manual' else 0), # Priority 1: Human Verified
                        (x['end'] - x['start'])                                               # Priority 2: Length
                    ), reverse=True)
                    
                    final_anns = []
                    for cand in valid_anns:
                         is_overlapping = False
                         cand_rng = set(range(cand['start'], cand['end']))
                         
                         for existing in final_anns:
                             exist_rng = set(range(existing['start'], existing['end']))
                             if cand_rng.intersection(exist_rng):
                                 is_overlapping = True
                                 break
                         
                         if not is_overlapping:
                             final_anns.append(cand)
                    
                    valid_anns = final_anns

                    # Tokenize & Tag (Basic whitespace tokenizer to match standard CoNLL)
                    # For robust CoNLL, we need to respect the tokenization.
                    # As a heuristic, we split by space and check alignment.
                    
                    # Skip empty text
                    if not text or not text.strip():
                        # st.warning(f"Empty text for SID {sid}") # DEBUG
                        continue

                    # DEBUG: Print first sentence text to log
                    if i == 0:
                        st.info(f"First exported sentence (SID {sid}): {text[:100]}...")

                    # SPACY TOKENIZATION
                    doc = nlp(text)
                    
                    for token in doc:
                        # Find token char offsets
                        start_idx = token.idx
                        end_idx = token.idx + len(token)
                        
                        # Determine BIO Tag
                        label = "O"
                        for a in valid_anns:
                            # Strict containment: token is inside annotation
                            if start_idx >= a['start'] and end_idx <= a['end']:
                                # Determine B- or I-
                                if start_idx == a['start']:
                                    label = f"B-{a['label']}"
                                else:
                                    label = f"I-{a['label']}"
                                break
                            
                            # Partial Overlap Handling (Token crosses boundary)
                            # e.g. token="Athens," ann="Athens" -> token indices include comma
                            elif a['start'] >= start_idx and a['end'] <= end_idx:
                                # Token contains the annotation
                                label = f"B-{a['label']}"
                                break
                            
                            # If annotation starts INSIDE token but ends OUTSIDE token (rare case for Spacy)
                            # e.g. token="don't" ann="n't"
                            elif start_idx < a['start'] and end_idx > a['start']:
                                # Overlaps start of token? No, overlaps END of token.
                                # Treat as B if it covers significant portion?
                                # Let's stick to strict or loose
                                if end_idx > a['start']:
                                     label = f"B-{a['label']}"
                                     break

                        f.write(f"{token.text} {label}\n")
                    
                    f.write("\n") # Sentence break
                    count_exported += 1
                
                    if i % 10 == 0:
                        prog_bar.progress((i + 1) / len(rows))
                
                prog_bar.progress(1.0)
            
            # --- STATISTICS GENERATION ---
            # Calculate Stats after file is written
            
            stats_md_path = full_path.replace(".conll", "_statistics.md")
                
            # Simple logic to parse the file we just wrote
            tag_counts = {}
            entity_counts = {}
            token_count = 0
            sent_count = 0
            
            current_sent_len = 0
            
            with open(full_path, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()
            
            # DEBUG: Print number of lines read
            if not lines:
                 st.error("Stats Generation: Read 0 lines from file!")
            else:
                 pass
                 # st.info(f"Stats Generation: Read {len(lines)} lines.")

            # Analyze
            current_entity_type = None
            current_entity_len = 0
            
            doc_entity_lengths = {} # type -> list of lengths
            
            for line in lines:
                line = line.strip()
                
                # Check for empty line (sentence boundary)
                if not line:
                    if current_sent_len > 0:
                        sent_count += 1
                        current_sent_len = 0
                    
                    # Reset entity tracking at sentence break
                    if current_entity_type:
                        if current_entity_type not in doc_entity_lengths: doc_entity_lengths[current_entity_type] = []
                        doc_entity_lengths[current_entity_type].append(current_entity_len)
                        current_entity_type = None
                        current_entity_len = 0
                    continue
                        
                token_count += 1
                current_sent_len += 1
                    
                parts = line.split()
                if len(parts) >= 2:
                    tag = parts[-1]
                else:
                    tag = "O"
                    
                # Tag Counts
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Entity Logic (BIO)
                if tag.startswith("B-"):
                    # Close previous if active
                    if current_entity_type:
                        if current_entity_type not in doc_entity_lengths: doc_entity_lengths[current_entity_type] = []
                        doc_entity_lengths[current_entity_type].append(current_entity_len)
                    
                    # Start new
                    etype = tag[2:]
                    entity_counts[etype] = entity_counts.get(etype, 0) + 1
                    current_entity_type = etype
                    current_entity_len = 1
                    
                elif tag.startswith("I-") and current_entity_type:
                    # Continue existing
                    # Check if type matches
                    etype = tag[2:]
                    if etype == current_entity_type:
                        current_entity_len += 1
                    else:
                        # Mismatch (Broken I- tag), treat as new start or loose text?
                        # Standard CoNLL: Treat as B- if type changes or I- without B-
                        # We just close old and start new count for stats
                        if current_entity_type not in doc_entity_lengths: doc_entity_lengths[current_entity_type] = []
                        doc_entity_lengths[current_entity_type].append(current_entity_len)
                        
                        current_entity_type = etype
                        entity_counts[etype] = entity_counts.get(etype, 0) + 1 # Count as new entity occurence? Or just malformed?
                        current_entity_len = 1
                        
                elif tag == "O":
                    if current_entity_type:
                        if current_entity_type not in doc_entity_lengths: doc_entity_lengths[current_entity_type] = []
                        doc_entity_lengths[current_entity_type].append(current_entity_len)
                        current_entity_type = None
                        current_entity_len = 0

            # Final check (for last sentence if file doesn't end with blank line)
            if current_sent_len > 0:
                sent_count += 1
            if current_entity_type:
                if current_entity_type not in doc_entity_lengths: doc_entity_lengths[current_entity_type] = []
                doc_entity_lengths[current_entity_type].append(current_entity_len)

            # Generate Markdown
            total_entities = sum(entity_counts.values())
            avg_ent_len = 0
            if total_entities > 0:
                all_lens = [l for lens in doc_entity_lengths.values() for l in lens]
                avg_ent_len = sum(all_lens) / len(all_lens)
            
            md_content = f"""# Dataset Statistics: {output_name}

## Overall Statistics
- **Total Sentences**: {sent_count}
- **Total Tokens**: {token_count}
- **Total Entities**: {total_entities}
- **Avg Entity Length (tokens)**: {avg_ent_len:.2f}

## Entity Counts by Type
| Entity Type   |   Count |
|:--------------|--------:|
"""
            # Sort by count desc
            for etype, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                md_content += f"| {etype:<13} | {count:>7} |\n"

            md_content += """
## Tag Counts (BIO)
| Tag    |   Count |
|:-------|--------:|
"""
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                    md_content += f"| {tag:<6} | {count:>7} |\n"

            md_content += """
## Average Length per Entity Type
| Entity Type   |   Avg Length |   Min |   Max |
|:--------------|-------------:|------:|------:|
"""
            for etype in sorted(doc_entity_lengths.keys()):
                lens = doc_entity_lengths[etype]
                avg_l = sum(lens) / len(lens) if lens else 0
                min_l = min(lens) if lens else 0
                max_l = max(lens) if lens else 0
                md_content += f"| {etype:<13} | {avg_l:>12.2f} | {min_l:>5} | {max_l:>5} |\n"
            
            # Write MD
            with open(stats_md_path, "w", encoding="utf-8") as f_md:
                f_md.write(md_content)
                
            if count_exported > 0:
                st.success(f"✅ Export Complete! Saved to: {full_path} ({count_exported} sentences)")
            else:
                st.warning(f"⚠️ Export finished but 0 sentences were written. Check if texts are empty.")
            
            st.info(f"📊 Statistics generated: {stats_md_path}")
            
            # Display Stats in App
            st.markdown("---")
            st.markdown(md_content)

        except Exception as e:
            st.error(f"Export failed: {e}")
            
    conn.close()
