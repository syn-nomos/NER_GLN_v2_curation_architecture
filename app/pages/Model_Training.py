import streamlit as st
import os
import sys
import pandas as pd
import glob
import json

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.append(ROOT_DIR)

from src.database.db_manager import DBManager
from src.training.trainer import ActiveTrainer

st.set_page_config(page_title="Active Learning Studio", page_icon="🧠", layout="wide")

if "db" not in st.session_state:
    st.warning("Please connect to a database first.")
    st.stop()
    
db = st.session_state.db

st.title("🧠 Active Learning Studio")
st.markdown("Fine-tune your RoBERTa model with the latest verified annotations.")

# --- Stats Section ---
# Get counts
conn = db._get_conn()
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM sentences WHERE status='annotated'")
total_annotated = c.fetchone()[0]

c.execute("SELECT COUNT(DISTINCT sentence_id) FROM annotations WHERE source='manual'")
total_verified = c.fetchone()[0]

# Estimates
st.metrics_row = st.columns(3)
st.metrics_row[0].metric("Verified Sentences", total_verified, help="Sentences with at least one manual verification")
st.metrics_row[1].metric("Total Annotated", total_annotated)
st.metrics_row[2].metric("Ready for Training", total_verified)
conn.close()

st.markdown("---")

# --- Configuration ---
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Model Configuration")
    
    # Base Model Selection
    default_base = "xlm-roberta-base"
    # Find local versions
    versions_dir = os.path.join(ROOT_DIR, "models", "versions")
    local_models = [d for d in glob.glob(os.path.join(versions_dir, "*")) if os.path.isdir(d)]
    local_models_names = [os.path.basename(p) for p in local_models]
    
    base_model_option = st.selectbox(
        "Base Model", 
        ["xlm-roberta-base"] + local_models_names,
        index=len(local_models_names) # Default to latest local if exists, else 0
    )
    
    if base_model_option == "xlm-roberta-base":
        model_path = "xlm-roberta-base"
    else:
        model_path = os.path.join(versions_dir, base_model_option)

    st.info(f"Using base: `{model_path}`")

    # Hyperparameters
    epochs = st.slider("Epochs", 1, 10, 3)
    batch_size = st.select_slider("Batch Size", [2, 4, 8, 16], value=4)
    lr = st.select_slider("Learning Rate", [1e-5, 2e-5, 5e-5], value=2e-5)

with c2:
    st.subheader("Labels Configuration")
    # Define the schema we want the model to learn
    # Typically this mirrors your DB labels + BIO tags
    available_labels = ["LEG-REFS", "ORG", "PERSON", "GPE", "DATE", "FACILITY", "LOCATION", "PUBLIC_DOCS"]
    selected_labels = st.multiselect("Active Labels", available_labels, default=available_labels)
    
    # Convert to BIO List
    bio_labels = ["O"]
    for l in selected_labels:
        bio_labels.append(f"B-{l}")
        bio_labels.append(f"I-{l}")
        
    st.caption(f"Map: {bio_labels}")

# --- Training Action ---
st.markdown("### Training Control")

if st.button("🚀 Start Retraining Loop", type="primary"):
    
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    status_box.markdown("⏳ **Step 1: Fetching Verified Data...**")
    
    # 1. Fetch Data
    # Get ID/TEXT for verified sentences
    # We join with annotations to ensure we only get ones that HAVE manual annotations
    conn = db._get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT s.id, s.text 
        FROM sentences s 
        JOIN annotations a ON s.id = a.sentence_id 
        WHERE a.source = 'manual'
    """)
    rows = c.fetchall()
    
    if len(rows) < 10:
        st.error(f"Not enough verified data to train (Found {len(rows)}, need at least 10). Go annotate more!")
        st.stop()
        
    # Construct Sentence Objects with their Annotations
    train_data = []
    for sid, text in rows:
        # Get anns
        c.execute("SELECT label, start_offset, end_offset, text, source FROM annotations WHERE sentence_id = ? AND source='manual'", (sid,))
        anns = []
        for r in c.fetchall():
            anns.append({
                "label": r[0],
                "start_offset": r[1],
                "end_offset": r[2],
                "text": r[3],
                "source": r[4]
            })
        train_data.append({
            "id": sid,
            "text": text,
            "annotations": anns
        })
    conn.close()
    
    progress_bar.progress(20)
    status_box.markdown(f"⏳ **Step 2: Initializing Trainer with {len(train_data)} sentences...**")
    
    try:
        # Initialize Trainer
        trainer = ActiveTrainer(
            base_model_path=model_path,
            output_dir=versions_dir,
            labels_list=bio_labels
        )
        
        status_box.markdown("⏳ **Step 3: Training in progress... (This may take a while)**")
        progress_bar.progress(40)
        
        # Train
        new_model_path, metrics = trainer.train(
            train_sentences=train_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr
        )
        
        progress_bar.progress(100)
        status_box.success(f"✅ Training Complete! Model saved to: `{os.path.basename(new_model_path)}`")
        
        st.json(metrics)
        st.balloons()
        
    except Exception as e:
        status_box.error(f"Training Failed: {e}")
        st.exception(e)

