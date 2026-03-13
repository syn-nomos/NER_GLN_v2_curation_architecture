import sqlite3  # Ensure sqlite3 is imported for the helper function
import streamlit as st
import os
import sys
import time

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
# Prioritize local src
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.database.db_manager import DBManager
# Background Fixer
from src.core.background_fixer import get_background_fixer
bg_fixer = get_background_fixer()

# LLM Fixer Imports
try:
    from src.judges.llm_client import LLMJudge
    from src.agents.specific_boundary_fixer import TypeSpecificBoundaryFixer
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Model Loader
from src.models.roberta_ner import RobertaNER

st.set_page_config(page_title="Annotation Studio", page_icon="📝", layout="wide")

@st.cache_resource
def load_roberta():
    # Attempt to find a robust model path
    candidates = [
        r"E:\ALGORITHMS\Semi-auto annotation model\to_move\src\agents\roberta_model\mdeberta-v3-base\seed_1",
        "src/agents/GreekLegalRoBERTa_New", # Newly trained model (Priority 1)
        "E:/ALGORITHMS/Semi-auto annotation model/src/agents/GreekLegalRoBERTa_New",
        "src/agents/GreekLegalRoBERTa_v3",
        "E:/ALGORITHMS/Semi-auto annotation model/src/agents/GreekLegalRoBERTa_v3",
        "xlm-roberta-base" # Fallback
    ]
    for path in candidates:
        if "xlm" in path or os.path.exists(path):
            try:
                print(f"Loading NER from: {path}")
                return RobertaNER(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    return None

ner_model = load_roberta()

# Initialize LLM Fixer if enabled
if HAS_LLM and "fixer" not in st.session_state:
    try:
        # Check for key in sidebar later or config
        judge = LLMJudge()
        if judge.api_key:
            st.session_state.fixer = TypeSpecificBoundaryFixer(judge)
            st.session_state.judge = judge # Save pure judge for Phase 2
        else:
            st.session_state.fixer = None
            st.session_state.judge = None
    except Exception:
        st.session_state.fixer = None
        st.session_state.judge = None

# --- AI COUNCIL / HYBRID PREDICTOR SETUP ---
from src.core.hybrid_predictor import HybridPredictor
from src.agents.leg_refs_regex_agent import LegRefsRegexAgent
from src.agents.gpe_regex_agent import GpeRegexAgent
from src.agents.org_regex_agent import OrgRegexAgent
from src.agents.date_regex_agent import DateRegexAgent
from src.agents.public_docs_regex_agent import PublicDocsRegexAgent
from src.core.augmented_embeddings import AugmentedEmbeddingBuilder

@st.cache_resource
def load_hybrid_system(_roberta_instance): # Underscore prevents hashing issues
    if not _roberta_instance: return None
    # Agents
    agents = [
        LegRefsRegexAgent(), 
        GpeRegexAgent(), 
        OrgRegexAgent(),
        DateRegexAgent(),
        PublicDocsRegexAgent()
    ]
    # Initialize Hybrid Brain
    return HybridPredictor(
        roberta_model=_roberta_instance,
        vector_memory=None, 
        attention_extractor=None,
        embedding_builder=AugmentedEmbeddingBuilder(),
        rule_agents=agents
    )

hybrid_brain = load_hybrid_system(ner_model)

# Initialize Session State
if "db" not in st.session_state:
    st.warning("Please load or select a database from the Data Loader page first.")
    st.stop()

db = st.session_state.db

# --- INIT MEMORY MANAGER ---
from src.core.memory_manager import MemoryManager
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager(db, ner_model)
    st.toast("🧠 Memory Manager Initialized", icon="🧠")

memory_manager = st.session_state.memory_manager

# SELF-HEALING: Check if MemoryManager is stale (missing new methods)
if not hasattr(memory_manager, 'rebuild_all_vectors'):
    st.toast("🔄 Refactoring Memory Intelligence...", icon="🛠️")
    import importlib
    import src.core.memory_manager
    importlib.reload(src.core.memory_manager)
    from src.core.memory_manager import MemoryManager
    st.session_state.memory_manager = MemoryManager(db, ner_model)
    memory_manager = st.session_state.memory_manager

# SELF-HEALING: Check if DB instance is stale (missing new methods)
if not hasattr(db, 'add_annotation'):
    st.warning("🔄 Upgrading Database Connector... (One-time reload)")
    # Re-initialize to pick up new code changes
    try:
        from src.database.db_manager import DBManager
        import importlib
        import src.database.db_manager
        importlib.reload(src.database.db_manager) # Force reload of module
        
        st.session_state.db = src.database.db_manager.DBManager(db.db_path)
        db = st.session_state.db
        st.rerun() # Updated from experimental_rerun
    except Exception as e:
        st.error(f"Failed to reload DB: {e}")

st.title("📝 Annotation Studio")
st.caption(f"Connected to: {os.path.basename(db.db_path)}")

with st.expander("ℹ️ How to use"):
    st.markdown("""
    - **Annotate Pending**: Focuses only on sentences that have not been annotated yet (skips existing ones).
    - **Review/Correct**: Allows browsing ALL sentences (including already annotated ones) to make corrections.
    - Use the sidebar filters to narrow down by dataset split (train/dev/test).
    """)

# Sidebar Controls
st.sidebar.header("Filter Settings")
mode = st.sidebar.radio("Work Mode", ["Annotate Pending", "Review/Correct"], index=1)

target_split = st.sidebar.multiselect("Dataset Split", ["train", "dev", "test", "pending"], default=["train", "dev", "test", "pending"])

st.sidebar.markdown("---")
# Memory moved to Active Learning > Tab 5
        
do_propagate = st.sidebar.checkbox("📡 Auto-Propagate Decisions", value=False, help="If checked, Accepting/Deleting an entity will automatically update identical pending entities.")

st.sidebar.markdown("---")
st.sidebar.header("Visual Filters")
st.sidebar.caption("Toggle visibility by source:")
show_manual = st.sidebar.checkbox("✅ Verified (Human)", value=True)
show_import = st.sidebar.checkbox("📄 Imported (CoNLL)", value=True)
show_auto = st.sidebar.checkbox("🤖 Suggestions (AI)", value=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📝 Notepad", expanded=False):
    # Robust path to data/notepad.txt
    try:
        data_root = os.path.dirname(ROOT_DIR) if os.path.basename(ROOT_DIR) == 'app' else ROOT_DIR
        nf_path = os.path.join(data_root, "data", "notepad.txt")
    except:
        nf_path = "notepad.txt" 
    
    # Load on first run
    if "notepad_content" not in st.session_state:
        if os.path.exists(nf_path):
            try:
                with open(nf_path, "r", encoding="utf-8") as f:
                    st.session_state.notepad_content = f.read()
            except:
                st.session_state.notepad_content = ""
        else:
            st.session_state.notepad_content = ""

    def save_pad():
        with open(nf_path, "w", encoding="utf-8") as f:
            f.write(st.session_state.notepad_content)
            
    st.text_area("Auto-saved notes:", key="notepad_content", height=200, on_change=save_pad)

# Helper to get sentences based on mode
@st.cache_data(ttl=60) # Cache for performance, invalidate often
def get_sentence_ids(db_path, mode, splits):
    """Get list of IDs matching criteria."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if mode == "Annotate Pending":
        query = f"SELECT id FROM sentences WHERE status = 'pending' AND dataset_split IN ({','.join(['?']*len(splits))}) ORDER BY id ASC"
        cursor.execute(query, splits)
    else:
        # Review Mode - Get all or annotated
        query = f"SELECT id FROM sentences WHERE dataset_split IN ({','.join(['?']*len(splits))}) ORDER BY id ASC"
        cursor.execute(query, splits)
        
    ids = [r[0] for r in cursor.fetchall()]
    conn.close()
    return ids

# Reload IDs if filters change
ids_pool = get_sentence_ids(db.db_path, mode, target_split)
total_count = len(ids_pool)

# Fetch Data logic updated for navigation
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# --- TABS LAYOUT ---
tab_annotate, tab_auto = st.tabs(["✍️ Workbench (Manual)", "🤖 Auto-Annotator (Council)"])

with tab_auto:
    st.markdown("### 🚀 Mass Annotation Pipeline")
    st.info("This tool scans the filtered sentences and applies AI annotations automatically. It skips sentences that already have annotations.")
    
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    
    with c1:
        st.markdown("**1. Strategy**")
        strategy = st.radio("Phase Selection", 
                           ["Phase 1: Local (Fast)", "Phase 2: Council (Cloud)", "Hybrid"], 
                           index=0, horizontal=True)
    
    with c2:
        st.markdown("**2. Batch**")
        batch_limit = st.number_input("Process max items:", 10, 5000, 50, help="How many NEW items to find and tag.")
    
    with c3:
        st.markdown("**3. Speed**")
        delay = st.number_input("Delay (s)", 0.0, 5.0, 0.1)
    
    with c4:
        st.markdown("**4. Conf**")
        min_conf = st.slider("Threshold", 0.0, 1.0, 0.5)
        
    if st.button("✨ START BATCH", disabled=(total_count==0), type="primary"):
        pbar = st.progress(0)
        logs = st.empty()
        st.toast("Pipeline started...")
        
        # LOGIC: Iterate through the filtered pool
        # But only process if NO annotations exist
        
        processed_count = 0
        skipped_count = 0
        
        # We walk through the pool until we hit the batch_limit of PROCESSED items
        # or we run out of items.
        
        current_batch = []
        
        for pid in ids_pool:
            if processed_count >= batch_limit:
                break
            
            s = db.get_sentence(pid)
            
            # CHECK: Resume Logic
            # If sentence already has annotations (from manual or previous run), skip it.
            if s.get('annotations') and len(s['annotations']) > 0:
                skipped_count += 1
                continue
            
            # Add to processing queue
            current_batch.append(s)
            
            # If queue full, process
            if len(current_batch) >= 10: # Mini-batch for model
                 # RUN MODEL
                 if "Phase 1" in strategy or "Hybrid" in strategy:
                    if hybrid_brain:
                        for item in current_batch:
                            preds = hybrid_brain.predict(item['text'])
                            for p in preds:
                                db.add_annotation(
                                    sentence_id=item['id'],
                                    label=p['label'],
                                    start_offset=p['start'],
                                    end_offset=p['end'],
                                    text_content=p['text'],
                                    source='auto'
                                )
                 
                 # Update counters
                 processed_count += len(current_batch)
                 current_batch = []
                 logs.text(f"Tagged {processed_count} sentences... (Skipped {skipped_count})")
                 pbar.progress(min(processed_count / batch_limit, 1.0))
                 time.sleep(delay)

        # Process remaining
        if current_batch:
             if "Phase 1" in strategy or "Hybrid" in strategy:
                if hybrid_brain:
                    for item in current_batch:
                        preds = hybrid_brain.predict(item['text'])
                        for p in preds:
                            db.add_annotation(
                                sentence_id=item['id'],
                                label=p['label'],
                                start_offset=p['start'],
                                end_offset=p['end'],
                                text_content=p['text'],
                                source='auto'
                            )
             processed_count += len(current_batch)
             pbar.progress(1.0)
        
        st.success(f"Batch Complete! ✅ Tagged: {processed_count}, Skipped (Already Done): {skipped_count}")
        time.sleep(1.5)
        st.rerun()

with tab_annotate:
    # --- MAIN SINGLE ITEM UI ---
    if total_count == 0:
        st.info("No sentences found matching criteria.")
    else:
        # Navigation
        # Ensure index in bounds
        if st.session_state.current_idx >= total_count:
            st.session_state.current_idx = 0
            
        current_id = ids_pool[st.session_state.current_idx]
        
        # Load Sentence
        sentence = db.get_sentence(current_id)



if total_count == 0:
    st.warning("No sentences found matching criteria.")
    st.stop()

# Progress Indicator
curr = st.session_state.current_idx + 1
st.progress(curr / total_count, text=f"Sentence {curr} of {total_count} ({mode})")

# Navigation Controls
col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])

with col_nav1:
    if st.button("⬅️ Previous"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)

with col_nav3:
    if st.button("Next ➡️"):
        st.session_state.current_idx = min(total_count - 1, st.session_state.current_idx + 1)

with col_nav2:
    # Direct Jump
    curr = st.session_state.current_idx + 1
    new_idx = st.number_input(f"Sentence {curr} / {total_count}", min_value=1, max_value=total_count, value=curr, label_visibility="collapsed")
    if new_idx != curr:
        st.session_state.current_idx = new_idx - 1
        st.rerun()

# Get Current ID
current_id = ids_pool[st.session_state.current_idx]
current_sent = db.get_sentence(current_id)

# --- EDITOR UI ---
st.markdown(f"### Sentence #{current_sent['id']}")

# Status Badge
status_color = "green" if current_sent['status'] == 'annotated' else "orange"
st.markdown(f"**Status:** :{status_color}[{current_sent['status'].upper()}]")

# --- RENDERER: Annotated Sentence (HTML Version) ---
def render_inline_annotations(text, annotations):
    """
    Renders text with inline annotations using HTML for rich styling.
    """
    # Wrapper style that defaults to current theme string color (supports dark/light mode naturally)
    wrapper_style = "font-family: sans-serif; font-size: 1.1em; line-height: 1.8; padding: 15px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.2); background-color: rgba(128, 128, 128, 0.05);"

    if not annotations:
        return f"<div style='{wrapper_style}'>{text}</div>"

    # Entities Colors (Hex)
    TYPE_COLORS = {
        "LEG-REFS": "#8A2BE2", # BlueViolet
        "ORG": "#1E90FF",      # DodgerBlue
        "PERSON": "#FF4500",   # OrangeRed
        "GPE": "#DAA520",      # GoldenRod
        "DATE": "#2E8B57",     # SeaGreen
        "FACILITY": "#20B2AA", # LightSeaGreen
        "LOCATION": "#CD853F", # Peru
        "PUBLIC-DOCS": "#708090" # SlateGray (fixed typo)
    }

    # Helper to convert Hex to RGBA for light background spans
    def hex_to_rgba(hex_color, alpha=0.15):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3: hex_color = ''.join([c*2 for c in hex_color])
        if len(hex_color) != 6: return f"rgba(128, 128, 128, {alpha})"
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    # 1. Sort by start index (reverse)
    sorted_anns = sorted(annotations, key=lambda x: x['start_offset'], reverse=True)
    
    # 2. Reconstruct
    processed_text = text
    
    for ann in sorted_anns:
        s, e = ann['start_offset'], ann['end_offset']
        lbl = ann['label']
        src = ann.get('source', 'unknown')
        
        # Determine Status Style & Apply correct typing background
        is_verified = ann.get('is_correct') == 1 or ann.get('is_accepted') == 1 or src == 'manual'
        status_icon = "✓" if is_verified else ("🤖" if src == 'auto' else "📥")
        
        # Determine Type Color
        type_hex = TYPE_COLORS.get(lbl, "#888888") # Default Gray
        
        bg = hex_to_rgba(type_hex, 0.15)
        border = type_hex
        
        # Safe slice content
        entity_text = text[s:e]
        
        # HTML Snippet: Color strictly matches tag, text does not force black anymore.
        html_snippet = (
            f"<span style='background-color: {bg}; border-bottom: 2px solid {border}; "
            f"padding: 2px 4px; border-radius: 4px; margin: 0 2px; white-space: nowrap;' title='Source: {src}'>"
            f"<strong>{entity_text}</strong> "
            f"<span style='background-color: {type_hex}; color: white; padding: 1px 4px; "
            f"border-radius: 3px; font-size: 0.75em; font-weight: bold; vertical-align: middle;'>{lbl} {status_icon}</span>"
            f"</span>"
        )
        
        # Replace
        if e <= len(text):
            processed_text = processed_text[:s] + html_snippet + processed_text[e:]
            
    return f"<div style='{wrapper_style}'>{processed_text}</div>"

# Use get_annotations_for_sentence instead of get_annotations
try:
    annotated_view = render_inline_annotations(current_sent['text'], db.get_annotations_for_sentence(current_sent['id']))
except AttributeError:
    # Fallback in case method name is different in some legacy DBManager version
    try:
        annotated_view = render_inline_annotations(current_sent['text'], db.get_annotations(current_sent['id']))
    except:
        annotated_view = render_inline_annotations(current_sent['text'], [])

st.markdown(annotated_view, unsafe_allow_html=True)
st.caption("Legend: 🟩 Verified (Human) | 🟦 Imported (CoNLL) | 🟪 Suggestion (AI)")
# --- EDIT BOUNDARIES / PULL WORDS ---
with st.expander("✂️ Fix Sentence Boundaries (Pull Words)"):
    st.info("Automatically pull words/tokens from adjacent sentences in the document. Annotations will safely move along with the text!")
    
    try:
        adj = db.get_adjacent_sentences(current_sent['id'], current_sent['source_doc'])
        
        col_pull1, col_pull2, col_pull3 = st.columns([1.2, 1.5, 1.2])
        
        with col_pull1:
            if adj.get('prev'):
                if st.button("⬅️ Pull Word from Previous", use_container_width=True, key="pull_prev"):
                    success = db.pull_word_from_previous_sentence(current_sent['id'], current_sent['source_doc'])
                    if success:
                        st.toast("Pulled from previous sentence!")
                        st.rerun()
                    else:
                        st.warning("Nothing left to pull.")
                st.caption(f"**Prev:** {adj['prev']['text']}")
            else:
                st.caption("No previous sentence in this document.")
                
        with col_pull2:
            st.markdown(f"**Current Text:**\n\n> {current_sent['text']}")
            
        with col_pull3:
            if adj.get('next'):
                if st.button("Pull Word from Next ➡️", use_container_width=True, key="pull_next"):
                    success = db.pull_word_from_next_sentence(current_sent['id'], current_sent['source_doc'])
                    if success:
                        st.toast("Pulled from next sentence!")
                        st.rerun()
                    else:
                        st.warning("Nothing left to pull.")
                st.caption(f"**Next:** {adj['next']['text']}")
            else:
                st.caption("No next sentence in this document.")
                
    except Exception as e:
        st.error(f"Error loading adjacent sentences: {e}")
# --- Global Sentence Actions ---
if HAS_LLM and st.session_state.fixer:
    if st.button("🤖 Ask LLM for Missing Entities", help="Query LLM in background for missed entities"):
        # We need existing annotations
        temp_anns = db.get_annotations_for_sentence(current_sent['id'])
        payload = {
            'type': 'missing',
            'sentence_id': current_sent['id'],
            'text_context': current_sent['text'],
            'existing_entities': [a['text'] for a in temp_anns]
        }
        bg_fixer.add_task(payload)
        st.toast("Background LLM started scanning for missing entities!", icon="🚀")

# Load existing annotations
annotations = db.get_annotations_for_sentence(current_sent['id'])

# --- Manual Add Section --
with st.expander("➕ Add Missing Entity (Manual)"):
    m_col1, m_col2 = st.columns([2, 1])
    
    # 1. Text Input (search)
    manual_text = m_col1.text_input("Entity Text", placeholder="Paste text here...", key="manual_text_input")
    manual_label = m_col2.selectbox("Label", ["LEG-REFS", "ORG", "PERSON", "GPE", "DATE", "FACILITY", "LOCATION", "PUBLIC_DOCS"], key="manual_label")

    if manual_text:
        # 2. Find Occurrences Logic
        import re
        # Escape special chars to treat as literal, but ignore case
        try:
            matches = [m for m in re.finditer(re.escape(manual_text), current_sent['text'], re.IGNORECASE)]
        except:
            matches = []

        if not matches:
            st.warning("⚠️ Text not found in sentence.")
        
        elif len(matches) == 1:
            # Single Match - Simple Add
            m = matches[0]
            st.info(f"Target found at position {m.start()}-{m.end()}")
            if st.button("Add Entity"):
                new_ann = {
                    "label": manual_label,
                    "text": current_sent['text'][m.start():m.end()], # Capture exact case from text
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "source": "manual",
                    "is_correct": 1
                }
                annotations.append(new_ann)
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                st.success("Added!")
                st.rerun()
                
        else:
            # Multiple Matches - Disambiguation UI
            st.info(f"Start found **{len(matches)} occurrences**. Please select which one:")
            
            # Create readable options
            options = {}
            for i, m in enumerate(matches):
                start, end = m.start(), m.end()
                # Create a small context snippet
                snippet_start = max(0, start - 20)
                snippet_end = min(len(current_sent['text']), end + 20)
                
                prefix = ("..." if snippet_start > 0 else "") + current_sent['text'][snippet_start:start]
                target = current_sent['text'][start:end]
                suffix = current_sent['text'][end:snippet_end] + ("..." if snippet_end < len(current_sent['text']) else "")
                
                # HTML formatted label for radio/selectbox wouldn't render HTML, so we use plain text logic or specialized
                options[f"Occurrence {i+1}"] = {
                    "match": m,
                    "display": f"#{i+1}: {prefix}[{target}]{suffix}"
                }
            
            selected_occ = st.radio("Select Occurrence:", list(options.keys()), 
                                    format_func=lambda x: options[x]['display'])
            
            if st.button("Add Selected Occurrence"):
                m = options[selected_occ]['match']
                new_ann = {
                    "label": manual_label,
                    "text": current_sent['text'][m.start():m.end()],
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "source": "manual",
                    "is_correct": 1
                }
                annotations.append(new_ann)
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                st.success("Added!")
                st.rerun()

st.markdown("#### Annotations")

# --- Background Missing Entities Results ---
fixes = bg_fixer.get_completed_fixes()
for fix in list(fixes):  # use list to allow removal during iteration if needed later
    if fix['type'] == 'missing' and fix['original'].get('sentence_id') == current_sent['id']:
        st.info(f"✨ **LLM Found Missing Entities:**\n" + "\n".join([f"- `{e.get('text')}` -> **{e.get('label')}**" for e in fix['found']]))
        nc1, nc2 = st.columns([1,1])
        if nc1.button("✔️ Add All", key=f"add_all_missing_{current_sent['id']}"):
            for e in fix['found']:
                # Find start/end offsets if possible. If not, just simple string match
                import re
                try:
                    m = re.search(re.escape(e['text']), current_sent['text'], re.IGNORECASE)
                    s_off, e_off = (m.start(), m.end()) if m else (0, len(e['text']))
                except:
                    s_off, e_off = (0, len(e['text']))
                    
                annotations.append({
                    'text': e['text'],
                    'label': e['label'],
                    'start_offset': s_off,
                    'end_offset': e_off,
                    'source': 'manual', # Accepted LLM is manual
                    'is_correct': 1
                })
            db.save_annotations(current_sent['id'], annotations, mark_complete=False)
            bg_fixer.completed_fixes.remove(fix)
            st.toast("Missing entities added!", icon="✅")
            st.rerun()
        if nc2.button("❌ Dismiss", key=f"dismiss_missing_{current_sent['id']}"):
            bg_fixer.completed_fixes.remove(fix)
            st.rerun()

if annotations:
    # Entity Color Map
    ENTITY_COLORS = {
        "LEG-REFS": "#8A2BE2", # BlueViolet
        "ORG": "#1E90FF",      # DodgerBlue
        "PERSON": "#FF4500",   # OrangeRed
        "GPE": "#DAA520",      # GoldenRod
        "DATE": "#2E8B57",     # SeaGreen
        "FACILITY": "#20B2AA", # LightSeaGreen
        "LOCATION": "#DAA520", # Same as GPE usually
        "PUBLIC_DOCS": "#4682B4" # SteelBlue
    }

    for i, ann in enumerate(annotations):
        source = ann.get('source', 'unknown')

        # Determine source type for filtering
        is_manual = source == 'manual'
        is_import = 'conll' in source or 'import' in source
        is_auto = 'model' in source or 'active' in source
        
        # Apply Sidebar Filters
        if is_manual and not show_manual: continue
        if is_import and not show_import: continue
        if is_auto and not show_auto: continue
        if (not is_manual and not is_import and not is_auto) and not show_import: continue # Fallback for unknown

        label = ann['label']
        text = ann['text']
        color = ENTITY_COLORS.get(label, "#6c757d") # Default Grey

        # Icon based on source
        if source == 'manual': icon = "✅" 
        elif 'conll' in source or 'import' in source: icon = "📄"
        elif 'model' in source or 'active' in source: icon = "🤖"
        else: icon = "❓"

        # HTML Badge Rendering
        html = f"""
        <div style="margin-bottom: 0px;">
            <span style="font-size: 1.2em; margin-right: 6px; vertical-align: middle;">{icon}</span>
            <span style="background-color: {color}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 500; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                {text}
                <span style="background-color: rgba(255,255,255,0.2); margin-left: 8px; padding: 1px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; font-family: sans-serif;">
                    {label}
                </span>
            </span>
            <small style="color: #888; margin-left: 8px;">{source}</small>
        </div>
        """
        
        # Layout: Badge | Actions
        # Adjusted columns for better alignment
        c1, c2 = st.columns([0.7, 0.3])
        
        with c1:
            st.markdown(html, unsafe_allow_html=True)
            
        with c2:
            # Action Buttons: [Edit] [Auto-Fix] [Accept?] [Delete]
            # Using smaller column widths inside c2 for the buttons
            cols = st.columns([1, 1, 1, 1])
            b_edit = cols[0]
            b_fix = cols[1]
            b_accept = cols[2]
            b_del = cols[3]
            
            # 1. Edit (Always available)
            if b_edit.button("✏️", key=f"edit_{ann.get('id', text)}_{i}", help="Manual Edit"):
                 st.session_state.editing_annotation = ann
                 st.rerun()

            # 2. Auto-Fix (LLM) - Background Task
            if HAS_LLM and st.session_state.fixer:
                if b_fix.button("✨", key=f"autofix_{ann.get('id', text)}_{i}", help="Auto-Fix Boundaries (LLM)"):
                    payload = {
                        'type': 'bounds',
                        'annotation_id': ann.get('id', text),
                        'sentence_id': current_sent['id'],
                        'text_context': current_sent['text'],
                        'current_span': text,
                        'label': label
                    }
                    bg_fixer.add_task(payload)
                    st.toast("Sent to background LLM! Continue working...", icon="🏃")

        # 5. Background LLM Results Notification Area
        # Check globally for any completed background fixes targeting this sentence
        fixes = bg_fixer.get_completed_fixes()
        for fix in fixes:
            if fix['original'].get('sentence_id') == current_sent['id'] and fix['original'].get('annotation_id') == ann.get("id", text):
                st.info(f"✨ **LLM Background Result (Ready):**\n\nOriginal: `{fix['original']['current_span']}`\n\nCorrected: `{fix['refined_span']}`")
                nc1, nc2 = st.columns([1,1])
                if nc1.button("✔️ Accept", key=f"bg_acc_{ann.get('id', text)}_{i}"):
                    ann['text'] = fix['refined_span']
                    ann['source'] = 'manual'
                    db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                    bg_fixer.completed_fixes.remove(fix)
                    st.toast("Fix applied successfully!", icon="✅")
                    st.rerun()
                if nc2.button("❌ Reject", key=f"bg_rej_{ann.get('id', text)}_{i}"):
                    bg_fixer.completed_fixes.remove(fix)
                    st.rerun()

        # 3. Accept (Only if not already manual)
        if not is_manual:
            if b_accept.button("✅", key=f"accept_{ann.get('id', text)}_{i}", help="Verify (Mark as Correct)"):
                ann['source'] = 'manual'
                ann['is_correct'] = 1
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                
                # Propagation Logic
                if do_propagate:
                    n_up = memory_manager.propagate(text, label, 'accept')
                    if n_up > 0: st.toast(f"Propagated ACCEPT to {n_up} other items.", icon="📡")
                
                st.rerun()
        
        # 4. Delete (Always available)
        if b_del.button("🗑️", key=f"del_{ann.get('id', text)}_{i}", help="Delete Annotation"):
            new_anns = [a for a in annotations if a != ann]
            db.save_annotations(current_sent['id'], new_anns, mark_complete=False)
            
            # Propagation Logic
            if do_propagate:
                n_up = memory_manager.propagate(text, label, 'reject')
                if n_up > 0: st.toast(f"Propagated REJECT to {n_up} other items.", icon="📡")
            
            st.rerun()

    # --- Editing Modal/Expander ---
    if "editing_annotation" in st.session_state and st.session_state.editing_annotation:
        edit_ann = st.session_state.editing_annotation
        with st.form(key="edit_form"):
            st.markdown(f"**Editing:** `{edit_ann['text']}`")
            new_label = st.selectbox("Label", list(ENTITY_COLORS.keys()), index=list(ENTITY_COLORS.keys()).index(edit_ann['label']) if edit_ann['label'] in ENTITY_COLORS else 0)
            
            # Simple text fix for now (Advanced boundary fixing requires token interaction)
            new_text = st.text_input("Text (Manual Fix)", value=edit_ann['text'])
            
            # --- Auto-Fix Button ---
            if HAS_LLM and st.session_state.fixer:
                if st.form_submit_button("✨ Auto-Fix Boundaries (LLM)"):
                    with st.spinner("🤖 Asking LLM to fix boundaries..."):
                        corrected = st.session_state.fixer.fix_boundary(current_sent['text'], new_text, new_label)
                        if corrected and corrected != new_text:
                            # We can't update the text_input directly without rerun or session state trickery
                            # Simplest: Update the edit_ann and rerun to refresh the modal state
                            st.session_state.editing_annotation['text'] = corrected
                            st.rerun()
                        else:
                            st.warning("LLM proposed no change.")

            c_edit1, c_edit2 = st.columns(2)
            if c_edit1.form_submit_button("💾 Save Changes"):
                # Update the annotation in the list
                # This is tricky without unique IDs if there are duplicates, but we try best match
                for i, a in enumerate(annotations):
                    if a == edit_ann:
                        annotations[i]['label'] = new_label
                        annotations[i]['text'] = new_text
                        annotations[i]['source'] = 'manual' # Edited implies manual
                        annotations[i]['is_correct'] = 1
                        break
                db.save_annotations(current_sent['id'], annotations, mark_complete=False)
                del st.session_state.editing_annotation
                st.rerun()
                
            if c_edit2.form_submit_button("❌ Cancel"):
                 del st.session_state.editing_annotation
                 st.rerun()

else:
    st.write("_No annotations yet._")

# --- Ask LLM Assistant (Manual Query) ---
st.divider()
if "llm_expanded" not in st.session_state:
    st.session_state["llm_expanded"] = False

def keep_llm_open():
    st.session_state["llm_expanded"] = True

# Clear LLM answer if sentence changes
if "llm_last_sentence_id" not in st.session_state:
    st.session_state["llm_last_sentence_id"] = current_sent['id']
elif st.session_state["llm_last_sentence_id"] != current_sent['id']:
    st.session_state["llm_last_sentence_id"] = current_sent['id']
    if "llm_answer" in st.session_state:
        del st.session_state["llm_answer"]

with st.expander("🤖 Ask LLM Assistant", expanded=st.session_state["llm_expanded"]):
    st.caption("Ask an AI about this sentence or a specific part of it.")
    
    # Model Selection
    available_models = {
        "Google: Gemini 2.5 Flash (Default)": "google/gemini-2.5-flash",
        "Anthropic: Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
        "DeepSeek: DeepSeek-R1": "deepseek/deepseek-r1",
        "xAI: Grok": "x-ai/grok-2",
        "OpenAI: GPT-4o Mini": "openai/gpt-4o-mini"
    }

    selected_model_name = st.selectbox("Select AI Model", list(available_models.keys()), index=0, key="llm_model_select")
    selected_model_id = available_models[selected_model_name]

    llm_mode = st.radio("Context Scope", ["Whole Sentence", "Specific Segment"], horizontal=True, key="llm_mode", on_change=keep_llm_open)

    llm_segment = None
    if llm_mode == "Specific Segment":
        lc1, lc2 = st.columns(2)
        with lc1:
            l_start = st.number_input("Start Index (LLM)", min_value=0, max_value=len(current_sent['text']), value=0, key="llm_start", on_change=keep_llm_open)
        with lc2:
            l_end = st.number_input("End Index (LLM)", min_value=0, max_value=len(current_sent['text']), value=len(current_sent['text']), key="llm_end", on_change=keep_llm_open)

        if l_end > l_start:
            llm_segment = current_sent['text'][l_start:l_end]
            st.info(f"Selected Segment: **{llm_segment}**")

    with st.form(key="llm_form"):
        user_question = st.text_area("Your Question", placeholder="π.χ. ειναι το επιλεγμένο κείμενο ORG ή εινια κατι άλλο;")
        submit_button = st.form_submit_button("🚀 Send Prompt", type="primary")

    if submit_button:
        st.session_state["llm_expanded"] = True
        if not user_question:
             st.warning("Please enter a question.")
        else:
            st.divider()
            judge = LLMJudge(model=selected_model_id)
            try:
                if hasattr(judge, 'stream_question'):
                    full_response = st.write_stream(judge.stream_question(current_sent['text'], user_question, llm_segment))
                else:
                    st.warning("Streaming not natively supported by this client instantiation, using simple call. (Restart app to fix)")
                    context_str = f"\nSegment: {llm_segment}" if llm_segment else ""
                    prompt = f"Sentence: {current_sent['text']}{context_str}\n\nQuestion: {user_question}"
                    # Check if _call_llm supports expect_json
                    import inspect
                    if 'expect_json' in inspect.signature(judge._call_llm).parameters:
                        res = judge._call_llm(system_prompt="You are an expert NLP annotator in Greek.", user_prompt=prompt, expect_json=False)
                    else:
                        res = judge._call_llm(system_prompt="You are an expert NLP annotator in Greek.", user_prompt=prompt)
                    
                    if isinstance(res, str):
                        full_response = res
                    elif isinstance(res, dict) and 'choices' in res:
                        full_response = res['choices'][0]['message']['content']
                    elif res:
                        full_response = str(res)
                    else:
                        full_response = "No response. (The model might have returned text while the app expected JSON)."
                        
                    st.write(full_response)
                
                st.session_state['llm_answer'] = full_response
            except Exception as e:
                st.error(f"Error calling LLM: {e}")

    if "llm_answer" in st.session_state and not submit_button:
        st.divider()
        st.success("### AI Response")
        st.markdown(st.session_state['llm_answer'])

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Confirm / Mark as Verified"):
        # Logic: Convert all current annotations to 'manual' source and mark sentence as annotated
        updated_annotations = []
        for ann in annotations:
            ann_copy = ann.copy()
            ann_copy['source'] = 'manual' # Upgrade to manual/verified
            updated_annotations.append(ann_copy)
        
        # Save back to DB
        db.save_annotations(current_sent['id'], updated_annotations, mark_complete=True)
        st.success("✅ Verified & Saved! Moving to next...")
        
        # Move to next
        st.session_state.current_idx = min(total_count - 1, st.session_state.current_idx + 1)
        st.rerun()

with col2:
    if st.button("✏️ Edit"):
        st.info("Editor logic pending implementation.")
