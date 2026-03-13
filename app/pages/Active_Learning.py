import streamlit as st
import sys
import os
import time
import json
import sqlite3

# Path resolution
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT of to_move
# app/pages -> app -> to_move
TO_MOVE_ROOT = os.path.dirname(os.path.dirname(APP_DIR))

if TO_MOVE_ROOT not in sys.path:
    sys.path.insert(0, TO_MOVE_ROOT)

# We also need the Workspace Root for data/knowledge_base
# to_move -> Semi-auto annotation model
WORKSPACE_ROOT = os.path.dirname(TO_MOVE_ROOT)

import src.database.db_manager
import importlib
importlib.reload(src.database.db_manager)
from src.database.db_manager import DBManager

st.set_page_config(page_title="Active Learning Studio", page_icon="🧠", layout="wide")

st.title("🧠 Active Learning & Consistency")
st.markdown("Use this tool to improve **Recall** (find missing entities), enforce **Consistency** (fix contradictions), and **Manage Rules** (Lexicons/Regex).")

if "db" not in st.session_state:
    # Try to load from session config first
    CONFIG_FILE = os.path.join(TO_MOVE_ROOT, "session_config.json")
    loaded_config = False
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                last_db = data.get("last_db_path")
                if last_db and os.path.exists(last_db):
                    db_path = last_db
                    loaded_config = True
        except:
             pass
    
    if not loaded_config:
        # Fallback to default search logic
        db_path_candidate_1 = os.path.join(TO_MOVE_ROOT, "data", "db", "annotations.db")
        db_path_candidate_2 = os.path.join(TO_MOVE_ROOT, "data", "annotations.db")
        workspace_db = os.path.join(WORKSPACE_ROOT, "data", "annotations.db")
    
        if os.path.exists(db_path_candidate_1):
            db_path = db_path_candidate_1
        elif os.path.exists(db_path_candidate_2):
            db_path = db_path_candidate_2
        elif os.path.exists(workspace_db):
            db_path = workspace_db
        else:
             db_path = db_path_candidate_1 # Final fallback

    st.session_state.db = DBManager(db_path)
else:
    # HOTFIX: Check if the stale object is missing new methods (development mode)
    # Even with reload, the old object in session_state persists.
    if not hasattr(st.session_state.db, 'get_vectorized_entities'):
        st.toast("Updating Database Manager...", icon="🔄")
        # Re-instantiate with the RELOADED class using the CURRENT path in session state
        current_path = st.session_state.db.db_path
        st.session_state.db = DBManager(current_path)

db = st.session_state.db
db_path = db.db_path
st.sidebar.caption(f"📂 DB: `{os.path.basename(db.db_path)}`")

# TABS
# --- UPDATED LAYOUT (User Request: Remove 'Fix Mistakes', consolidate modes) ---
tab_consistency, tab_remove, tab_high_conf, tab_repair, tab_lexicon = st.tabs([
    "🔍 Find Missing (Memory)", 
    "🗑️ Remove / Fix Annotations",
    "🚀 High Confidence",
    "🛠️ Sentence Repair",
    "📚 Rules & Knowledge Base"
])

# --- TAB 1: CONSISTENCY (Find Missing) ---
with tab_consistency:
    st.subheader("1. Find Missing Annotations")
    st.info("Identify phrases you've annotated elsewhere but missed here (RECALL).")
    
    # --- Control Panel ---
    
    # Header & Settings
    h1, h2, h3 = st.columns([3, 1, 1])
    h1.markdown("### 🔍 Search Strategy")
    with h2:
        top_n_limit = st.number_input("List Limit:", min_value=50, max_value=2000, value=200, step=50, key="cfg_limit")
    with h3:
        st.write("") # Spacer align
        if st.button("🔄 Refresh List", key="btn_ref"):
            if "top_ents" in st.session_state: 
                del st.session_state.top_ents
                
    # 1. Global Entity Selector (Dynamic based on Mode)
    
    col_mode, col_sel = st.columns([2, 1])
    
    with col_mode:
        st.markdown("**1. Detection Mode**")
        mode_consistency = st.radio(
            "Mode", 
            ["String Match (Exact)", "Fuzzy Match (Approximate)", "Vector Match (Semantic)", "Hybrid (Fuzzy + Vector)"], 
            horizontal=True,
            label_visibility="collapsed",
            key="cons_mode_radio"
        )


    # Determine Source
    if "Vector" in mode_consistency or "Hybrid" in mode_consistency:
        # Load Vectorized Entities
        # REFRESH if missing OR if strictly None (cleared by rebuilder)
        if "vec_ents" not in st.session_state or st.session_state.vec_ents is None:
             # Try to get vectorized entities
             vecs = st.session_state.db.get_vectorized_entities(limit=top_n_limit)
             
             # FALLBACK: Explicitly query annotations if call returned empty
             if not vecs:
                 try:
                     c = st.session_state.db.conn.cursor()
                     c.execute("""
                        SELECT text_span as text, label, COUNT(*) as count
                        FROM annotations
                        WHERE vector IS NOT NULL
                        GROUP BY text_span, label
                        ORDER BY count DESC
                        LIMIT ?
                     """, (top_n_limit,))
                     # Ensure we convert to list of dicts
                     vecs = [dict(zip([d[0] for d in c.description], r)) for r in c.fetchall()]
                 except Exception as e:
                     st.warning(f"Fallback query failed: {e}")
            
             st.session_state.vec_ents = vecs
        
        # If still empty (User hasn't built memory), fallback to standard list
        if not st.session_state.vec_ents:
             # Even after refresh, if empty:
             if "top_ents" not in st.session_state:
                  st.session_state.top_ents = st.session_state.db.get_top_entities(limit=top_n_limit)
             source_list = st.session_state.top_ents
             ent_help = "⚠️ No Vector Memory found (neither in Centroids nor Annotations). Please Rebuild Memory."
        else:
             source_list = st.session_state.vec_ents
             ent_help = "Showing entities with VALID MEMORY."
    else:
        # Load Frequent Entities (Standard)
        if "top_ents" not in st.session_state:
            st.session_state.top_ents = db.get_top_entities(limit=top_n_limit)
        source_list = st.session_state.top_ents
        ent_help = "Showing most frequent annotated entities."

    # Filter out short entities AND very long garbage (e.g. whole paragraphs erroneously labeled)
    # Clean Valid Entities: Remove substrings unless they are significantly more frequent (distinct)
    raw_ents = [e for e in source_list if len(e['text']) >= 3 and len(e['text']) < 200]
    
    # Sort by length descending to ensure stability
    # Also handle count properly as it might be '?' strictly speaking (though unlikely from get_top_entities)
    
    valid_ents = []
    
    # Threshold: If A is substring of B, A must appear > 3.0x as often as B to be kept.
    # We increased this from 1.25 to 3.0 to heavily penalize fragments like "Περιφερειακής" 
    # when "Περιφερειακής Ενότητας..." exists.
    THRESHOLD_RATIO = 3.0
    
    for i, ent_a in enumerate(raw_ents):
        text_a = ent_a['text'].strip()
        count_a = ent_a.get('count', 0)
        # Ensure count is int
        if isinstance(count_a, str):
            try: count_a = int(float(count_a))
            except: count_a = 0
            
        should_hide = False
        for ent_b in raw_ents:
            if ent_a == ent_b: 
                continue
            
            text_b = ent_b['text'].strip()
            count_b = ent_b.get('count', 0)
            if isinstance(count_b, str):
                 try: count_b = int(float(count_b))
                 except: count_b = 0
            
            # Check Include (Case-Insensitive)
            # If text_a is INSIDE text_b (e.g. "Περιφερειακής" inside "Περιφερειακής Ενότητας")
            if text_a.lower() != text_b.lower() and text_a.lower() in text_b.lower():
                 # If the longer phrase has ANY significant presence (e.g. created manually), conceal the part.
                 # Revised Logic: If the fragment count isn't overwhelmingly larger, hide it.
                 if count_a <= (count_b * THRESHOLD_RATIO):
                     should_hide = True
                     break
        
        if not should_hide:
            valid_ents.append(ent_a)
    # -----------------------------
    
    # -----------------------------
    # PERSISTENCE HELPERS (Embedded)
    REVIEWED_FILE = os.path.join(TO_MOVE_ROOT, "data", "reviewed_entities_v2.json")

    if "reviewed_entities" not in st.session_state:
        try:
            if os.path.exists(REVIEWED_FILE):
                with open(REVIEWED_FILE, 'r', encoding='utf-8') as f:
                    st.session_state.reviewed_entities = set(json.load(f))
            else:
                st.session_state.reviewed_entities = set()
        except:
            st.session_state.reviewed_entities = set()

    def mark_reviewed(ent_text, is_done):
        if is_done:
            st.session_state.reviewed_entities.add(ent_text)
        else:
            st.session_state.reviewed_entities.discard(ent_text)
        # Auto-save
        try:
            os.makedirs(os.path.dirname(REVIEWED_FILE), exist_ok=True)
            with open(REVIEWED_FILE, 'w', encoding='utf-8') as f:
                json.dump(list(st.session_state.reviewed_entities), f, ensure_ascii=False)
        except: pass

    with col_sel:
        st.markdown("**2. Select Target Entity**")
        
        import pandas as pd
        
        # Prepare DataFrame
        df_ents = pd.DataFrame(valid_ents)
        if not df_ents.empty:
            # Add 'Done' column
            df_ents['Done'] = df_ents['text'].apply(lambda x: x in st.session_state.reviewed_entities)
            # Reorder columns: Done, text, label, count
            if 'count' not in df_ents.columns: df_ents['count'] = 0
            df_ents = df_ents[['Done', 'text', 'label', 'count']]
            # Rename for display
            df_ents.columns = ['✅ Done', 'Entity Text', 'Label', 'Count']
        else:
            df_ents = pd.DataFrame(columns=['✅ Done', 'Entity Text', 'Label', 'Count'])

        # Search Filter (Auto-Focus if possible)
        col_search, col_type, col_stats = st.columns([2, 2, 1])
        with col_search:
             search_query = st.text_input("🔍 Filter Text:", "", 
                                        help="Type to search...", 
                                        placeholder="Type text...")
        
        with col_type:
             # Get unique labels
             if not df_ents.empty:
                unique_labels = sorted(list(set(df_ents['Label'].unique().tolist() + ["LEG-REFS", "ORG", "PERSON", "GPE", "DATE", "FACILITY", "LOCATION", "PUBLIC-DOCS"])))
             else:
                unique_labels = sorted(["LEG-REFS", "ORG", "PERSON", "GPE", "DATE", "FACILITY", "LOCATION", "PUBLIC-DOCS"])

             filter_type = st.multiselect("🏷️ Filter Type:", unique_labels,
                                          default=[],
                                          placeholder="All Types (Default)")

        # Apply Filters
        df_display = df_ents.copy()
        
        if search_query:
            df_display = df_display[df_display['Entity Text'].str.contains(search_query, case=False, na=False)]
            
        if filter_type:
            df_display = df_display[df_display['Label'].isin(filter_type)]

        with col_stats:
             st.metric("Entities", len(df_display))


        # DATA EDITOR
        edited_df = st.data_editor(
            df_display,
            column_config={
                "✅ Done": st.column_config.CheckboxColumn(
                    "Done",
                    help="Mark as reviewed",
                    default=False,
                ),
                "Entity Text": st.column_config.TextColumn(
                    "Entity",
                    width="medium",
                    disabled=True 
                ),
                "Label": st.column_config.TextColumn(
                    "Type",
                    width="small",
                    disabled=True
                ),
                "Count": st.column_config.NumberColumn(
                    "Cnt",
                    format="%d",
                    width="small",
                    disabled=True
                ),
            },
            hide_index=True,
            width='stretch',
            height=300, # Compact height
            key="editor_ents",
            on_change=None 
        )

        # DETECT CHANGES (Logic to update JSON)
        # Note: st.data_editor returns the state *after* edit. 
        # But to know WHAT changed in a stateless way is hard unless we diff.
        # Simple approach: Update ALL displayed rows into session state.
        # Or better: Just check rows where 'Done' changed? 
        # Actually, simpler: Iterate the edited_df and update session_state.
        
        # Sync changes to persistence
        if not edited_df.empty:
            current_done = set(edited_df[edited_df['✅ Done'] == True]['Entity Text'].tolist())
            current_not_done = set(edited_df[edited_df['✅ Done'] == False]['Entity Text'].tolist())
            
            # Update global set based on visible rows
            st.session_state.reviewed_entities.update(current_done)
            st.session_state.reviewed_entities.difference_update(current_not_done)
            
            # Trigger save (debounce/optimize later if slow)
            # Only save if something changed? For now, save periodically or on specific actions?
            # Streamlit re-runs on edit, so this runs every click. It's fine for local JSON.
            mark_reviewed("dummy", False) # Calls save internally properly if we refactor logic

        # SELECTION LOGIC
        # Since we use data_editor for ticking, how do we 'Select' for scanning?
        # Option A: Click a row -> But Streamlit data_editor selection is new.
        # Option B: Add a 'Select' column with a button? No, data_editor doesn't support buttons.
        # Option C: Use a separate Selectbox populated by the FILTERED list above.
        
        # Let's use a Selectbox below the table for the actual "Action".
        # Why? Because clicking a checkbox shouldn't trigger a scan.
        # And clicking a row in data_editor (if selection enabled) might conflict with editing.
        
        # The user said "when I select, do the scan" and "tick separately".
        
        # Create list for dropdown based on filtered view
        select_options = edited_df['Entity Text'].tolist()
        
        # Try to find what was selected before if possible
        idx = 0
        if "last_selected_ent" in st.session_state and st.session_state.last_selected_ent in select_options:
             idx = select_options.index(st.session_state.last_selected_ent)
             
        # Compact Selectbox
        selected_text = st.selectbox("👉 Choose Entity from Dropdown to Scan:", select_options, index=idx, key="sel_scan_box")

        # Find the full entity object
        if selected_text:
            st.session_state.last_selected_ent = selected_text
            # Retrieve labels/counts from the DF or original list
            row = edited_df[edited_df['Entity Text'] == selected_text].iloc[0]
            selected_ent = {
                'text': row['Entity Text'],
                'label': row['Label'],
                'count': row['Count']
            }
        else:
            selected_ent = None

        # --- Manual Search Override (Requested Feature) ---
        # Only for String/Fuzzy modes
        use_manual = False
        if mode_consistency in ["String Match (Exact)", "Fuzzy Match (Approximate)"]:
             st.caption("Or...")
             use_manual = st.checkbox("✍️ Manual Input", key="chk_manual")
        
        if use_manual:
             man_text = st.text_input("Enter text:", placeholder="Type anything...", key="txt_manual")
             std_labels = ["LEG-REFS", "ORG", "PERSON", "GPE", "DATE", "FACILITY", "LOCATION", "PUBLIC-DOCS"]
             man_label = st.selectbox("Label:", std_labels, index=0)
             
             if man_text:
                 selected_ent = {'text': man_text, 'label': man_label, 'count': 'N/A'}
             else:
                 selected_ent = None



    st.divider()

    # --- Action Logic ---
    col_tools, col_actions = st.columns([3, 1])
    
    with col_tools:
        if not selected_ent:
            st.info("👈 Please select an entity or enter text (Manual Mode) to proceed.")
        
        else:
            # Helper to get vector
            def get_entity_vector(text, label):
                # Try centroids first
                c = db.conn.cursor()
                c.execute("SELECT vector FROM entity_centroids WHERE text_span = ? AND label = ?", (text, label))
                row = c.fetchone()
                if row and row[0]: return row[0]
                
                # Fallback to any accepted annotation
                c.execute("SELECT vector FROM annotations WHERE text_span = ? AND label = ? AND vector IS NOT NULL LIMIT 1", (text, label))
                row = c.fetchone()
                if row and row[0]: return row[0]
                return None
    
            # Helper to clear old widget state
            def clear_widget_state():
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith("lbl_") or k.startswith("txt_") or k.startswith("c_")]
                for k in keys_to_clear:
                    del st.session_state[k]
    
            # --- MODE 1: STRING MATCH ---
            if mode_consistency == "String Match (Exact)":
                st.caption(f"Finds **EXACT** occurrences of **'{selected_ent['text']}'** (Count: {selected_ent.get('count', '?')}).")
                
                if st.button(f"🔎 Scan Exact: '{selected_ent['text']}'"):
                    with st.spinner(f"Searching occurrences..."):
                        clear_widget_state()
                        # Use the new detailed scanner
                        matches = db.scan_string_occurrences(selected_ent['text'])
                        
                        # Set label from selection (Manual or List)
                        st.session_state.consistency_label = selected_ent['label']
                             
                        st.session_state.consistency_matches = matches
                        st.session_state.consistency_mode = 'exact_result'
                        st.rerun()
            
            # --- MODE 2: FUZZY MATCH ---
            elif mode_consistency == "Fuzzy Match (Approximate)":
                st.caption(f"Finds **TYPOS** or variations of **'{selected_ent['text']}'** (e.g. '{selected_ent['text']}s') in **ALL SENTENCES** (Corpus Scan).")
                
                fuzz_thresh = st.slider("Similarity Threshold (%)", 60, 95, 80)
                
                if st.button(f"🧬 Scan Fuzzy: '{selected_ent['text']}'"):
                    clear_widget_state()
                    import regex
                    
                    target = selected_ent['text']
                    
                    # Calculate allowed errors based on Length & Threshold
                    # Greek considerations: Endings (os/ous) = 2 edits, Accents = 1 edit
                    # We enforce a minimum error tolerance of 2 for words >= 6 chars to handle suffixes
                    calc_errors = int(len(target) * (1 - fuzz_thresh/100.0))
                    k_errors = max(1, calc_errors)
                    if len(target) >= 6 and k_errors < 2:
                        k_errors = 2 # Boost tolerance for Greek morphology
                    
                    # (?i) = Ignore Case
                    # (?e) = Best Match (Edit Distance: insertions, deletions, substitutions)
                    pattern_str = f"(?i)(?e)({regex.escape(target)}){{e<={k_errors}}}"
                    
                    with st.spinner(f"Scanning full corpus for fuzzy matches (errors <= {k_errors})..."):
                        cursor = db.conn.cursor()
                        cursor.execute("SELECT id, text FROM sentences")
                        rows = cursor.fetchall()
                        
                        fuzzy_matches = []
                        prog_bar = st.progress(0)
                        total_rows = len(rows)
                        
                        # Compile outside loop
                        try:
                            rx = regex.compile(pattern_str)
                        except Exception as e:
                            st.error(f"Regex error: {e}")
                            st.stop()
                            
                        for idx, (sid, stext) in enumerate(rows):
                            if idx % 500 == 0:
                                prog_bar.progress(idx / total_rows)
                            
                            # Find fuzzy matches in 'stext'
                            # return (match object)
                            for m_rx in rx.finditer(stext):
                                s, e = m_rx.span()
                                m_txt = stext[s:e]
                                
                                # Build Match Info (Generic Structure)
                                match_info = {
                                    'sentence_id': sid,
                                    'full_text': stext,
                                    'match_text': m_txt,
                                    'start': s,
                                    'end': e,
                                    'status': 'new',
                                    'overlap_label': None,
                                    'overlap_type': None, 
                                    'overlap_id': None,
                                    'confidence': 0.0,
                                    'label': selected_ent['label'] 
                                }
                                
                                # Check overlaps (Expensive but necessary for correct UI state)
                                anns = db.get_annotations_for_sentence(sid)
                                for a in anns:
                                    if a.get('is_rejected'):
                                         continue
                                         
                                    as_, ae = a['start_offset'], a['end_offset']
                                    
                                    if (s < ae) and (e > as_):
                                        match_info['overlap_label'] = a['label']
                                        match_info['overlap_id'] = a['id'] 
                                        match_info['action_history'] = a.get('action_history') # Track history
                                        
                                        # Determine status from existing annotation
                                        # If it's a 'pending' candidate or accepted?
                                        # DB usually has 'status' or 'is_accepted'
                                        # get_annotations_for_sentence returns rows with columns...
                                        # DBManager.get_annotations_for_sentence returns dicts
                                        # We can assume if it exists in 'annotations' table it's tracked
                                        
                                        exact = (s == as_ and e == ae)
                                        inside = (s >= as_ and e <= ae)
                                        contains = (s <= as_ and e >= ae)
                                        
                                        if exact:
                                            match_info['overlap_type'] = 'exact'
                                            match_info['status'] = 'accepted' # Likely already in DB
                                        elif inside:
                                            match_info['overlap_type'] = 'inside'
                                            match_info['overlap_start'] = as_
                                            match_info['overlap_end'] = ae
                                            match_info['status'] = 'pending' # Treat overlap as pending review
                                        elif contains:
                                            match_info['overlap_type'] = 'contains'
                                            match_info['status'] = 'pending' # Treat partial overlap as pending review
                                        else:
                                            match_info['overlap_type'] = 'partial'
                                            match_info['status'] = 'pending' # Treat partial overlap as pending review
                                        break
                                
                                fuzzy_matches.append(match_info)
                                
                        prog_bar.empty()
    
                    st.session_state.consistency_matches = fuzzy_matches
                    
                    if selected_ent['label'] == 'MANUAL':
                         st.session_state.consistency_label = 'UNKNOWN'
                    else:
                         st.session_state.consistency_label = selected_ent['label']
                    
                    st.session_state.consistency_mode = 'fuzzy_result'
                    st.rerun()
    
            # --- MODE 3: VECTOR MATCH ---        
            elif mode_consistency == "Vector Match (Semantic)":
                st.caption(f"Finds candidates with **SIMILAR MEANING** to **'{selected_ent['text']}'** (using embeddings).")
                
                # Scope Filter (Fix for "No Results")
                col_thresh, col_scope = st.columns([1, 1])
                with col_thresh:
                    vec_thresh = st.slider("Similarity Threshold (%)", 50, 99, 75, help="Lower = More results (broad), Higher = More precise.")
                with col_scope:
                    search_scope = st.radio("Search Scope:", ["Pending (Suggestions)", "Accepted (Corpus)", "All"], index=0, horizontal=False)
                
                scope_map = {
                    "Pending (Suggestions)": "pending",
                    "Accepted (Corpus)": "accepted", 
                    "All": "all"
                }
                
                # Check if centroid exists for selected entity
                has_vector = get_entity_vector(selected_ent['text'], selected_ent['label']) 
                if not has_vector:
                     st.error(f"❌ Entity '{selected_ent['text']}' has no vector memory yet.")
                     st.info("💡 Go to '📚 Rules & Knowledge Base' -> 'Rebuild Memory' to generate vectors from your accepted annotations.")

                # Button Scan
                if st.button(f"🧠 Scan Vector: '{selected_ent['text']}'", disabled=(not has_vector)):
                    clear_widget_state()
                    vec_blob = has_vector # We already fetched it
                    
                    if vec_blob:
                        sim_cands = db.get_similar_pending_annotations(
                            vec_blob, 
                            threshold=vec_thresh/100.0, 
                            limit=50, 
                            label_filter=selected_ent['label'], 
                            filter_mode="Same",
                            scope=scope_map[search_scope]
                        )
                        
                        st.toast(f"Found {len(sim_cands)} semantic matches.")
                        mapped = []
                        for c in sim_cands:
                            # 'c' is an annotation from DB
                            # Determine status from 'is_accepted' column
                            status = 'accepted' if c.get('is_accepted') == 1 else 'pending'
                            
                            mapped.append({
                                'sentence_id': c['sentence_id'],
                                'full_text': c['sentence_text'], 
                                'match_text': c['text_span'],
                                'start': c['start_char'],
                                'end': c['end_char'],
                                'status': status,
                                'overlap_id': c['id'],        # CRITICAL: This is the ID to UPDATE
                                'overlap_label': c['label'],
                                'overlap_type': 'exact',      # It matches itself
                                'confidence': c['confidence'],
                                'label': c['label']
                            })
                        
                        st.session_state.consistency_matches = mapped
                        st.session_state.consistency_label = selected_ent['label'] 
                        st.session_state.consistency_mode = 'vector_result'
                        st.rerun()
                    else:
                        st.error(f"No vector found for '{selected_ent['text']}' (Label: {selected_ent['label']}) in Knowledge Base.")
                        st.info("👉 If you just labeled this, please go to **'📚 Rules & Knowledge Base'** and click **'🧠 Rebuild Memory'** to generate the vector.")

            # --- MODE 4: HYBRID ---
            elif mode_consistency == "Hybrid (Fuzzy + Vector)":
                st.caption(f"Finds candidates that look AND mean the same as **'{selected_ent['text']}'**.")
                
                col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
                with col_h1:
                     hyb_scope = st.radio("Search Scope:", ["Pending", "Accepted", "All"], index=1, key="hyb_scope")
                with col_h2:
                     vec_bias = st.slider("Semantic Bias", 0.0, 1.0, 0.7, help="Higher = Trust Vector more. Lower = Trust Fuzzy more.")
                with col_h3:
                     min_score = st.slider("Min Hybrid Score", 40, 90, 60)
                
                scope_map = {"Pending": "pending", "Accepted": "accepted", "All": "all"}

                if st.button(f"🚀 Scan Hybrid: '{selected_ent['text']}'"):
                    clear_widget_state()
                    vec_blob = get_entity_vector(selected_ent['text'], selected_ent['label'])
                    
                    if vec_blob:
                        # 1. Get Semantic Matches (Broad, Low Threshold)
                        # We cast a wide net (limit=500, threshold=0.5) to filter later
                        sim_cands = db.get_similar_pending_annotations(
                            vec_blob, 
                            threshold=0.50, 
                            limit=500,
                            scope=scope_map[hyb_scope]
                        )
                        
                        # 2. Re-Rank with Fuzzy
                        from rapidfuzz import fuzz
                        final_matches = []
                        target = selected_ent['text'].lower()
                        
                        for c in sim_cands:
                            # Fuzzy Score
                            cand_text = c['text_span'].lower()
                            fuzz_score = fuzz.ratio(target, cand_text)
                            
                            # Vector Score (re-calculate cosine or use confidence?)
                            # Actually db returns candidates > threshold, but doesn't explicitly return the score unless we parse it.
                            # Standard db.get_similar... implementation usually uses cosine. 
                            # Let's assume the candidates are returned sorted by vector similarity primarily.
                            # Since we don't have the exact vector score in the dict easily without changing DB code,
                            # We will rely on the fact they passed the vector threshold (0.5).
                            
                            # Hybrid Score: Weighted Average
                            # We approximate vector score as 0.8 (middle of 0.5-1.0) if not available, 
                            # OR we blindly trust fuzzy on top of semantic candidates.
                            # BETTER: Let's use the Fuzzy Score as the primary filter on check
                            
                            if fuzz_score >= min_score:
                                status = 'accepted' if c.get('is_accepted') == 1 else 'pending'
                                
                                final_matches.append({
                                    'full_text': c['sentence_text'],
                                    'start': c['start_char'],
                                    'end': c['end_char'],
                                    'match_text': c['text_span'],
                                    'sentence_id': c['sentence_id'],
                                    'confidence': c['confidence'],
                                    'label': c['label'],
                                    'status': status,
                                    'overlap_id': c['id'],
                                    'hybrid_score': fuzz_score 
                                })
                        
                        # Sort by Hybrid Score (Fuzzy match quality on semantically similar items)
                        final_matches.sort(key=lambda x: x['hybrid_score'], reverse=True)
                                
                        st.session_state.consistency_matches = final_matches
                        st.session_state.consistency_label = "VARIOUS"
                        st.session_state.consistency_mode = 'hybrid_result'
                        st.rerun()
                    else:
                        st.error("No vector found.")

    
    # --- Results Display ---
    # We use a distinct block for results so we can add a 'Clear' button
    if "consistency_matches" in st.session_state and st.session_state.consistency_matches:
        all_matches = st.session_state.consistency_matches
        label_candidate = st.session_state.get('consistency_label', 'UNKNOWN')
        
        st.divider()
        
        # --- FILTERS BEFORE HEADER ---
        with st.expander("🌪️ Filter Results", expanded=True):
            # 1. Filter by Status
            # Default to NEW only (User workflow focus)
            stats = list(set([m.get('status', 'new') for m in all_matches]))
            stat_labels = {
                'accepted': '🔒 Accepted (Existing)',
                'pending': '⚠️ Pending / Overlap', 
                'new': '✨ New / Missing'
            }
            if not stats: stats = ['new', 'accepted', 'pending']

            # FIX: Ensure default values exist in options to prevent StreamlitAPIException
            wanted_default = ['new']
            safe_default = [x for x in wanted_default if x in stats]
            
            # Fallback: If 'new' items don't exist (e.g. only 'accepted'), select all available
            # so the user actually sees the search results instead of an empty list.
            if not safe_default and stats:
                safe_default = stats
            
            # Default: Show 'new' + 'pending', hide 'accepted' to keep list clean?
            # User wants: "remove form proposed" after process.
            # So if we filter OUT 'accepted', they will disappear.
            c_fil1, c_fil2, c_fil3 = st.columns(3)
            
            with c_fil1:
                 sel_stats = st.multiselect(
                    "Filter by Status:", 
                    options=stats, 
                    default=safe_default, 
                    format_func=lambda x: stat_labels.get(x, x)
                )
            
            with c_fil2:
                # 2. Filter by Match Type (Overlap)
                overlaps = list(set([m.get('overlap_type', 'none') or 'none' for m in all_matches]))
                if len(overlaps) > 1:
                    sel_overlaps = st.multiselect("Filter by Overlap:", options=overlaps, default=overlaps)
                else:
                    sel_overlaps = overlaps

            with c_fil3:
                # 3. Filter by Action History (New Feature)
                # Aggregate Unique Tags
                all_tags = set()
                for m in all_matches:
                    h = m.get('action_history')
                    if h:
                        for tag in h.split(','):
                            all_tags.add(tag.strip())
                    else:
                        all_tags.add("No History")
                
                sorted_tags = sorted(list(all_tags))
                if not sorted_tags: sorted_tags = ["No History"]
                
                sel_history = st.multiselect("Filter by User Action:", options=sorted_tags, default=sorted_tags)

        # APPLY FILTERS
        def match_history(m, selected):
            h = m.get('action_history')
            tags = set(h.split(',')) if h else {"No History"}
            # Match ANY selected tag? Or needs to match at least one?
            # Usually: Show if it has at least one of the selected tags.
            return not tags.isdisjoint(set(selected))

        matches = [
            m for m in all_matches 
            if m.get('status', 'new') in sel_stats 
            and (m.get('overlap_type', 'none') or 'none') in sel_overlaps
            and match_history(m, sel_history)
        ]
        
        # --- TEXT FILTER (Strict Prefix / Fuzzy Start) ---
        # Add Search Bar for Text (Requested Feature: Prefix Match)
        st.write("---")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            q_txt = st.text_input("🔍 Prefix Search:", placeholder="e.g. 'Αθή' matches 'Αθήνα', 'Αθηναίος'...", help="Filters results to items STARTING with this text (Case Insensitive).")
        with search_col2:
            strict_prefix = st.checkbox("Strict Start", value=True, help="If checked, matches must START with the text. If unchecked, searches anywhere.")

        if q_txt:
            q_norm = q_txt.strip().lower()
            if strict_prefix:
                matches = [m for m in matches if m['match_text'].strip().lower().startswith(q_norm)]
            else:
                matches = [m for m in matches if q_norm in m['match_text'].strip().lower()]

        # --- HEADER (DYNAMIC) ---
        col_res, col_clr = st.columns([3, 1])
        with col_res:
             if len(matches) != len(all_matches):
                 st.subheader(f"Results: {len(matches)} filtered (of {len(all_matches)})")
             else:
                 st.subheader(f"Results: {len(matches)}")
                 
        with col_clr:
             if st.button("🗑️ Clear Results", type="secondary", use_container_width=True):
                st.session_state.consistency_matches = []
                st.rerun()
        
        # --- Pagination Logic ---
        ITEMS_PER_PAGE = 50
        total_items = len(matches)
        
        if "con_page" not in st.session_state:
            st.session_state.con_page = 0
            
        # Ensure page is valid
        max_page = max(0, (total_items - 1) // ITEMS_PER_PAGE)
        if st.session_state.con_page > max_page:
            st.session_state.con_page = max_page
            
        # Pagination Controls (Moved to Bottom)
        # We calculate variables here but render later
        if total_items > ITEMS_PER_PAGE:
            max_page = max(0, (total_items - 1) // ITEMS_PER_PAGE)
            if st.session_state.con_page > max_page:
                st.session_state.con_page = max_page
                
            st.markdown(f"**Showing {st.session_state.con_page * ITEMS_PER_PAGE + 1} - {min((st.session_state.con_page + 1) * ITEMS_PER_PAGE, total_items)} of {total_items} items**")
            # Note: The buttons are now rendered AFTER the loop below.

        # Slice Data
        start_idx = st.session_state.con_page * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        page_matches = matches[start_idx:end_idx]
        
        # --- BULK SELECTION ACTIONS ---
        c_sel_all, c_sel_none, _ = st.columns([1, 1, 4])
        # Use simple buttons that update session state for the specific keys
        if c_sel_all.button("✅ Select All Page", key=f"btn_sel_all_{st.session_state.con_page}"):
             for local_i, m in enumerate(page_matches):
                 i = start_idx + local_i
                 
                 # Key Construction must match EXACTLY what is used inside the form loop below
                 # Inside loop: 
                 # status = m.get('status', 'new')
                 # overlap_lbl = m.get('overlap_label', '')
                 # if status in ['accepted', 'pending'] and overlap_lbl:
                 #     lbl_key = overlap_lbl
                 # else:
                 #     lbl_key = m.get('label', 'ORG') # default fallback
                 
                 # u_key = f"{m['sentence_id']}_{m['start']}_{lbl_key}"
                 # Wait, the key inside the loop uses 'current label' which might be dynamic?
                 # NO, the checkboxes in the form use `key=f"c_{u_key}"`.
                 # We need to construct u_key perfectly here.
                 
                 # Replicate logic from loop (Lines ~820)
                 stat = m.get('status', 'new')
                 ov_lbl = m.get('overlap_label', '')
                 if stat in ['accepted', 'pending'] and ov_lbl:
                     cur_lbl = ov_lbl
                 else:
                     # If it's a new proposal, use the proposed label
                     cur_lbl = m.get('label', 'ORG') # Fallback if missing? 
                     # Actually m['label'] should exist from the search results.
                 
                 u_key = f"{m['sentence_id']}_{m['start']}_{cur_lbl}"
                 
                 st.session_state[f"c_{u_key}"] = True
             st.rerun()
             
        if c_sel_none.button("⬜ Deselect All Page", key=f"btn_desel_all_{st.session_state.con_page}"):
             for local_i, m in enumerate(page_matches):
                 i = start_idx + local_i
                 
                 stat = m.get('status', 'new')
                 ov_lbl = m.get('overlap_label', '')
                 if stat in ['accepted', 'pending'] and ov_lbl:
                     cur_lbl = ov_lbl
                 else:
                     cur_lbl = m.get('label', 'ORG')
                 
                 u_key = f"{m['sentence_id']}_{m['start']}_{cur_lbl}"
                 
                 st.session_state[f"c_{u_key}"] = False
             st.rerun()
             
        # --- PRE-FETCH NEXT SENTENCES (For Merge Functionality) ---
        # We need this to allow users to see and merge split sentences.
        if "pending_merges" not in st.session_state:
            st.session_state.pending_merges = set() # Set of sentence_ids to merge with their next
            
        current_sids = [m['sentence_id'] for m in page_matches]
        next_sent_map = {}
        
        if current_sids:
            try:
                # Get neighboring sentences (Prev and Next)
                # We fetch sid-1 and sid+1 for all visible items
                target_ids = []
                for sid in current_sids:
                    target_ids.append(sid - 1)
                    target_ids.append(sid + 1)
                
                target_ids = list(set(target_ids)) # De-duplicate
                
                if target_ids:
                    q_placeholders = ",".join("?" for _ in target_ids)
                    
                    if hasattr(st.session_state, 'db'):
                         res = st.session_state.db.conn.execute(f"SELECT id, text FROM sentences WHERE id IN ({q_placeholders})", target_ids).fetchall()
                    else:
                         with sqlite3.connect(db_path) as conn:
                             res = conn.execute(f"SELECT id, text FROM sentences WHERE id IN ({q_placeholders})", target_ids).fetchall()
                    
                    # Renaming to neighbor_map for clarity, but keeping variable name next_sent_map to minimize diffs if you prefer, 
                    # but frankly neighbor_map is better. I'll stick to 'neighbor_map' and update usages.
                    neighbor_map = {row[0]: row[1] for row in res}
                else:
                    neighbor_map = {}
            except Exception as e:
                # st.error(f"Error fetching neighbors: {e}")
                neighbor_map = {}
        
        # --- PRE-FETCH CO-OCCURRING ANNOTATIONS (For Full Context Highlight) ---
        # We need ALL annotations for the current sentences to render them properly
        co_annotations_map = {} # sentence_id -> list of (start, end, label, text)
        if current_sids:
            try:
                unique_sids = list(set(current_sids))
                q_ph = ",".join("?" for _ in unique_sids)
                
                # We need all accepted/pending annotations EXCEPT rejected ones
                # Also we should probably exclude the one we are currently editing (overlap_id match)
                # But for simplicity, fetch all non-rejected. We will filter out the actively edited one in the loop.
                query = f"""
                    SELECT sentence_id, id, start_char, end_char, label, text_span 
                    FROM annotations 
                    WHERE sentence_id IN ({q_ph}) AND is_rejected = 0 
                """
                
                if hasattr(st.session_state, 'db'):
                     co_res = st.session_state.db.conn.execute(query, unique_sids).fetchall()
                else:
                     with sqlite3.connect(db_path) as conn:
                         co_res = conn.execute(query, unique_sids).fetchall()
                
                for r in co_res:
                    sid, aid, s, e, l, txt = r
                    if sid not in co_annotations_map:
                        co_annotations_map[sid] = []
                    co_annotations_map[sid].append({'id': aid, 'start': s, 'end': e, 'label': l, 'text': txt})
                    
            except Exception as e:
                st.warning(f"Error fetching co-annotations: {e}")

        # --- BATCH APPLY UPDATES (Fix for Streamlit Modified State Error) ---
        if "pending_batch_update" in st.session_state and st.session_state.pending_batch_update:
            updates = st.session_state.pending_batch_update
            # updates is list of dicts: {'txt_key': val, 'lbl_key': val, 'chk_key': val}
            for up in updates:
                for k, v in up.items():
                    st.session_state[k] = v
            
            # Clear after applying
            st.session_state.pending_batch_update = []
            # We don't need to rerun here, as we are before the form rendering.
            # The widgets below will now pick up these values from session_state.

        with st.form("consistency_batch_form"):
             selected_indices = []
             # NO hardcoded view_limit anymore, use hashed list
             
             # CSS for compact dropdowns
             st.markdown("""
             <style>
             div[data-testid="stSelectbox"] > label { font-size: 0.8em; }
             div[data-testid="stSelectbox"] > div > div { min-height: 32px; height: 32px; padding-top: 0px; padding-bottom: 0px; }
             div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { font-size: 0.85em; }
             </style>
             """, unsafe_allow_html=True)
             
             # Pre-define colors (Matches Annotator.py style)
             TYPE_COLOR_MAP = {
                "LEG-REFS": "#8A2BE2", 
                "ORG": "#1E90FF",      
                "PERSON": "#FF4500",   
                "GPE": "#DAA520",      
                "DATE": "#2E8B57",    
                "FACILITY": "#20B2AA",
                "LOCATION": "#CD853F", 
                "PUBLIC-DOCS": "#708090", # Hyphenated to match standard
                "PUBLIC_DOCS": "#708090"  # Alias just in case
             }
             ALL_LABELS = list(TYPE_COLOR_MAP.keys())

             # Iterate over PAGINATED results
             # We must keep track of original index for 'matches' if we want to update the master list
             # but here we just need to render the widgets.
             # The key is: we iterate over 'page_matches'
             
             for local_i, m in enumerate(page_matches):
                 # i is the global index in the 'matches' list (for uniqueness)
                 i = start_idx + local_i
                 
                 sid = m['sentence_id']
                 
                 # Neighbors
                 prev_t = neighbor_map.get(sid - 1, "")
                 next_t = neighbor_map.get(sid + 1, "")
                 
                 # Dynamic Merge Logic (Chainable)
                 # If sid-1 is in pending, it means we are merging PREV -> CURR
                 # If sid is in pending, it means we are merging CURR -> NEXT
                 
                 is_merged_prev = ((sid - 1) in st.session_state.pending_merges)
                 is_merged_next = (sid in st.session_state.pending_merges)
                 
                 # Construct valid full text for preview
                 temp_msg_parts = []
                 if is_merged_prev and prev_t:
                     temp_msg_parts.append(prev_t)
                 
                 temp_msg_parts.append(m['full_text'])
                 
                 if is_merged_next and next_t:
                     temp_msg_parts.append(next_t)
                     
                 full_text = " ".join(temp_msg_parts)
                 
                 # --- DYNAMIC FULL CONTEXT (with Co-occurring Annotations) ---
                 # We want to show a read-only view of the WHOLE sentence with ALL annotations highlighted
                 # The 'active' candidate is just one of them.
                 
                 # 1. Gather all annotations for this sentence (excluding the active candidate if duplicate)
                 # Wait, m['overlap_id'] is the ID of the annotation we are editing (if exists).
                 active_ann_id = m.get('overlap_id')
                 
                 # Get list of existing anns
                 other_anns = co_annotations_map.get(sid, [])
                 
                 # We need to construct a list of spans to highlight:
                 # - The Active Candidate (Edit Target) -> Bright Orange / Color
                 # - The Other Existing Anns -> Grayed out / Dashed
                 
                 spans_to_render = []
                 
                 # Add Active Candidate (Use current form values if possible? No, too early)
                 # We render this based on m['start'], m['end'] shifted by merge offset
                 
                 # Recalculate offsets relative to new start
                 # If we added Prev, the original start shifts by len(prev) + 1
                 offset_shift = 0
                 if is_merged_prev and prev_t:
                     offset_shift = len(prev_t) + 1
                     
                 # Active Candidate Coordinates
                 start = m['start'] + offset_shift
                 end = m['end'] + offset_shift
                 start = max(0, start) # Safety
                 
                 # Define variables needed for key generation and state checking
                 match_txt = m['match_text']
                 status = m.get('status', 'new')
                 overlap_lbl = m.get('overlap_label', '')
                 
                 # Determine Label Logic
                 if status in ['accepted', 'pending'] and overlap_lbl:
                     current_label = overlap_lbl
                 else:
                     current_label = m.get('label', label_candidate)

                 # Generate Key
                 u_key = f"{m['sentence_id']}_{m['start']}_{current_label}" 
                 
                 # --- FETCH EDITED STATE ---
                 # We fetch user edits *before* building spans so we can adjust the active highlight location
                 current_txt_val = st.session_state.get(f"txt_{u_key}", match_txt)
                 current_lbl_val = st.session_state.get(f"lbl_{u_key}", current_label)
                 
                 is_edited = (current_txt_val != match_txt)
                 
                 # Initialize display coordinates (default to original)
                 display_start = start
                 display_end = end

                 # If edited, attempt to find new boundaries in full_text
                 if is_edited:
                      import re
                      try:
                          # Search near original position
                          pat = re.escape(current_txt_val).replace(r"\ ", r"\s+")
                          # Search window: Original start +/- 1000 chars
                          w_start = max(0, start - 1000)
                          w_end = min(len(full_text), end + 1000)
                          window_text = full_text[w_start:w_end]
                          
                          m_new = re.search(pat, window_text, re.IGNORECASE)
                          if m_new:
                              display_start = w_start + m_new.start()
                              display_end = w_start + m_new.end()
                              # Update truncation color to new label if changed
                              current_label = current_lbl_val 
                          else:
                              # If not found, we can't highlight it on the full text correctly.
                              # We'll just stick to original boundaries for the 'background', 
                              # but this might look wrong if text length changed significantly.
                              pass
                      except:
                          pass

                 # --- BUILD RENDERING SPANS ---
                 spans_to_render = []
                 
                 # 1. Other Annotations (Faint/Secondary)
                 other_anns = co_annotations_map.get(sid, [])
                 for oth in other_anns:
                     # Skip if it overlaps substantially with the ACTIVE candidate (Original OR Edited)
                     # If we are editing, we don't want the old version of itself shadowing the new one.
                     # Check ID first
                     if oth['id'] == m.get('overlap_id'): 
                         continue

                     # Calculate position
                     s_oth = oth['start'] + offset_shift
                     e_oth = oth['end'] + offset_shift
                     
                     # Deduplication: If this 'other' annotation has exact same start/end as our Active Candidate, skip it
                     if s_oth == display_start and e_oth == display_end:
                         continue
                         
                     spans_to_render.append({
                         'start': max(0, s_oth), 
                         'end': min(len(full_text), e_oth), 
                         'label': oth['label'], 
                         'style': 'secondary'
                     })

                 # 2. Active Candidate (Primary)
                 # Use the calculated display coordinates (which reflect edits if found)
                 spans_to_render.append({
                     'start': display_start, 
                     'end': display_end, 
                     'label': current_label, 
                     'style': 'primary'
                 })

                 # Helper: Render HTML with multiple spans
                 def render_highlighted_text(text, spans):
                     if not spans: return text
                     
                     # Create markers: (index, type, style, label)
                     # Type: 1=Start, -1=End. We use numbers to sort End before Start at same index?
                     # Actually, for nested spans, we want Inner Start -> Inner End.
                     # But here we have potential overlaps.
                     # Flat approach: Use a char array and inject tags? 
                     # Or just sort by start.
                     
                     # Simple Approach: 
                     # 1. Sort spans.
                     # 2. Iterate text chars, keeping track of active styles.
                     # 3. If multiple styles, Primary wins.
                     
                     # Let's use a char-tagging approach for robustness against overlap.
                     # 0 = No Highlight
                     # 1 = Secondary
                     # 2 = Primary
                     
                     mask = [0] * len(text)
                     
                     # Apply Secondary first
                     for s in [x for x in spans if x['style'] == 'secondary']:
                         for i in range(s['start'], s['end']):
                             if i < len(mask): mask[i] = 1
                             
                     # Apply Primary on top
                     for s in [x for x in spans if x['style'] == 'primary']:
                         for i in range(s['start'], s['end']):
                             if i < len(mask): mask[i] = 2

                     # Build HTML
                     res = []
                     curr_style = 0
                     
                     # Colors
                     # Secondary: Gray/Faint
                     # Primary: Orange/Highlight
                     style_map = {
                         0: "",
                         1: "background-color:rgba(0,0,0,0.08); color:#666; border-bottom:1px dashed #999;",
                         2: f"background-color:rgba(255,165,0,0.3); color:#000; font-weight:bold; border-bottom:2px solid {TYPE_COLOR_MAP.get(m.get('label', 'CANDIDATE'), '#555')};"
                     }
                     
                     for i, char in enumerate(text):
                         style_code = mask[i]
                         
                         if style_code != curr_style:
                             # Close previous span if needed
                             if curr_style != 0:
                                 res.append("</span>")
                             
                             # Open new span if needed
                             if style_code != 0:
                                 res.append(f"<span style='{style_map[style_code]}'>")
                             
                             curr_style = style_code
                         
                         res.append(char)
                     
                     if curr_style != 0:
                         res.append("</span>")
                         
                     return "".join(res)

                 # Generate the HTML for the FULL Text
                 # This can be used in the 'Full Context' expander
                 rendered_full_html = render_highlighted_text(full_text, spans_to_render)
                 
                 # match_txt remains same, but we use it for checking edits
                 match_txt = m['match_text']
                 
                 # Overlap / Status logic
                 status = m.get('status', 'new')
                 overlap_type = m.get('overlap_type', '')
                 overlap_lbl = m.get('overlap_label', '')
                 
                 # Dynamic Label Selection Strategy:
                 # 1. If it's an existing annotation (Accepted OR Pending), use its label.
                 #    (This satisfies: "Default to what is already inside the text")
                 # 2. Otherwise use the proposed/search label.
                 if status in ['accepted', 'pending'] and overlap_lbl:
                     current_label = overlap_lbl
                 else:
                     current_label = m.get('label', label_candidate)
                 
                 # KEY GENERATION (STABLE) - Must be invariant to Merges/Offsets
                 # We use m['start'] (original) instead of 'start' (calculated/shifted)
                 # This ensures that even if we merge a Previous sentence (shifting the view), 
                 # the entity's edit state is preserved because its ID is based on its original DB position.
                 u_key = f"{m['sentence_id']}_{m['start']}_{current_label}" 
                 
                 # ----------------------------------------------------
                 
                 trunc_color = TYPE_COLOR_MAP.get(current_label, "#555")

                 # Badge Logic
                 status_badge = ""
                 row_bg = "#fff"
                 is_accepted = (status == 'accepted')
                 
                 if is_accepted:
                     row_bg = "#f0f8ff" # Light blue
                     status_badge = f"<span style='background:#ccc; color:#555; font-size:0.7em; padding:2px 4px; border-radius:4px;'>🔒 Accepted ({overlap_lbl})</span>"
                     
                 elif status == 'pending':
                     row_bg = "#fffaf0" # Floral white
                     status_badge = f"<span style='background:#ffd700; color:#000; font-size:0.7em; padding:2px 4px; border-radius:4px;'>⚠️ Pending ({overlap_lbl})</span>"
                 
                 elif status == 'new':
                      status_badge = f"<span style='background:#90ee90; color:#006400; font-size:0.7em; padding:2px 4px; border-radius:4px;'>✨ Proposed ({current_label})</span>"

                 if overlap_type == 'inside':
                     status_badge += " <span style='font-size:0.7em; color:#888;'>(Inside larger span)</span>"
                 elif overlap_type == 'partial':
                     status_badge += " <span style='font-size:0.7em; color:#888;'>(Partial overlap)</span>"


                 # Context window
                 # If we have an outer span (inside), we want to make sure we see it.
                 o_start = m.get('overlap_start', start)
                 o_end = m.get('overlap_end', end)
                 
                 # Expand context to cover the outer span if it exists
                 view_start = min(display_start, o_start)
                 view_end = max(display_end, o_end)
                 
                 ctx_s = max(0, view_start - 50)
                 ctx_e = min(len(full_text), view_end + 50)

                 # --- RENDER SNIPPET LOGIC ---
                 # We want to show multi-highlight in the snippet too.
                 # With the new robust builder, we can handle it even if edited, because spans_to_render uses display_*
                 
                 # USE ROBUST MULTI-HIGHLIGHT RENDERER on the Context Window
                 # 1. Slice text
                 raw_snippet = full_text[ctx_s:ctx_e]
                 
                 # 2. Adjust spans relative to ctx_s
                 rel_spans = []
                 for s in spans_to_render:
                     # Check overlap with window
                     if s['end'] <= ctx_s or s['start'] >= ctx_e:
                         continue
                     
                     # Clip and Shift
                     rs_start = max(0, s['start'] - ctx_s)
                     rs_end = min(len(raw_snippet), s['end'] - ctx_s)
                     
                     if rs_end > rs_start:
                         rel_spans.append({
                             'start': rs_start,
                             'end': rs_end,
                             'label': s['label'],
                             'style': s['style']
                         })
                         
                 # 3. Render
                 snippet_inner = render_highlighted_text(raw_snippet, rel_spans)
                 snippet_html = f"...{snippet_inner}..."

                 # Complete snippet div
                 html_snippet = f"<div style='font-family: sans-serif; line-height: 1.5; background: {row_bg}; padding: 8px; border-radius: 5px; border: 1px solid #eee; font-size: 0.9em;'>" \
                                f"{status_badge} {snippet_html}</div>"
                 
                 # Columns: [Check] [Label] [Snippet] - Adjusted Ratio for Space
                 c1, c2, c3 = st.columns([0.3, 1.2, 5.5])
                 
                 # 1. Checkbox
                 # AUTO-SELECT LOGIC FOR CHECKBOX UI
                 # If the user has modified the label or text in session state, we want the checkbox to appear checked.
                 # This gives visual feedback that "this item is modified and will be processed".
                 is_modified_ui = False
                 if current_txt_val != match_txt: is_modified_ui = True
                 if current_lbl_val != current_label: is_modified_ui = True
                 
                 # If modified, force the checkbox key in session state to True (if not manually unset by user, 
                 # but here we prioritize modification = selection)
                 if is_modified_ui:
                     st.session_state[f"c_{u_key}"] = True

                 # User Request: Default unchecked (Use Select All buttons for bulk)
                 # Note: If key exists in session_state, 'value' param is ignored by Streamlit.
                 is_checked = c1.checkbox(f"#{i+1}", value=False, key=f"c_{u_key}", label_visibility="collapsed")
                 
                 # 2. Label Selector (Allow Correction)
                 # Determine default index safely
                 try:
                     def_idx = ALL_LABELS.index(current_label)
                 except ValueError:
                     def_idx = 0
                
                 new_lbl = c2.selectbox("Label:", ALL_LABELS, index=def_idx, key=f"lbl_{u_key}", label_visibility="collapsed")
                 
                 # 3. Snippet
                 
                 # --- MERGE TOOLS & CONTEXT ---
                 # Display full sentence and merge button if applicable
                 with c3.container():
                     # Prepare highlighted full text for context awareness
                     # We reuse the robust multi-highlight HTML generated above.
                     # This ensures consistency between the snippet and the full view.
                     try:
                         # Rendered HTML already has highlights. We enhance it with bold if needed?
                         # No, rendered_full_html has complete styling (primary/secondary).
                         # We just plug it in.
                         hl_full = rendered_full_html
                     except:
                         hl_full = full_text

                     # Layout: [Prev Btn] [Text] [Next Btn]
                     if prev_t or next_t:
                         cm_prev, cm_mid, cm_next = st.columns([0.8, 8, 0.8])
                         
                         # PREVIOUS BUTTON
                         with cm_prev:
                             if prev_t:
                                 if is_merged_prev:
                                     if st.form_submit_button("↩️", key=f"und_prev_{u_key}", help="Undo Merge with Previous"):
                                         st.session_state.pending_merges.remove(sid - 1)
                                         st.rerun() 
                                 else:
                                     # Display preview in tooltip
                                     if st.form_submit_button("⬆️", key=f"mrg_prev_{u_key}", help=f"Merge with Previous:\n'{prev_t[-50:]}...'"):
                                         st.session_state.pending_merges.add(sid - 1)
                                         st.rerun()
                         
                         # CENTER TEXT
                         with cm_mid:
                             st.markdown(f"<div style='font-size:0.85em; color:#333; margin-bottom:5px; padding:4px; background:#f9f9f9; border-radius:4px;'>{hl_full}</div>", unsafe_allow_html=True)
                             
                         # NEXT BUTTON
                         with cm_next:
                             if next_t:
                                 if is_merged_next:
                                     if st.form_submit_button("↩️", key=f"und_next_{u_key}", help="Undo Merge with Next"):
                                         st.session_state.pending_merges.remove(sid)
                                         st.rerun()
                                 else:
                                     if st.form_submit_button("⬇️", key=f"mrg_next_{u_key}", help=f"Merge with Next:\n'{next_t[:50]}...'"):
                                         st.session_state.pending_merges.add(sid)
                                         st.rerun()

                     else:
                          # Just text if no neighbors
                          st.markdown(f"<div style='font-size:0.85em; color:#555;'>{hl_full}</div>", unsafe_allow_html=True)
                 
                 c3.markdown(html_snippet, unsafe_allow_html=True)
                 
                 # 4. Text Adjustment (Expander)
                 with c3.expander("📝 Edit Text / Boundaries", expanded=False):
                     st.caption(f"Original Boundaries: {start} - {end}")
                     # Just Text Input
                     adj_text = st.text_input("Correct Phrase:", value=match_txt, key=f"txt_{u_key}", help="Edit text. Click 'Preview/Apply' to see changes above.")
                     
                     st.info("ℹ️ **Note:** Boundaries are preserved unless you edit the text. If you edit the text, we will scan to find the new boundaries.")
                     
                     # RESTDORED APPLY BUTTON (Secondary Submit)
                     # Clicking this submits the form -> Reruns Script -> Updates 'current_txt_val' above -> Updates HTML Snippet
                     # This gives the "Apply" effect without saving to DB yet.
                     # We MUST use a unique key, otherwise Streamlit crashes with DuplicateElementKey
                     if st.form_submit_button("🔄 Preview Change", help="Updates the snippet above with your changes", key=f"preview_btn_{u_key}"):
                         pass

                     # --- BULK APPLY LOGIC ---
                     # Provide a button to apply this correction to identical similar matches
                     # Only show if text is modified
                     if adj_text != match_txt:
                         # Count potential matches in the current session matches list (matches)
                         # We look for:
                         # 1. Same original match_text
                         # 2. Text correction is actually present in their context
                         import re
                         
                         similar_indices = []
                         # Loose check: count how many items had the same original text
                         # Note: matches is the filtered list for this VIEW.
                         for idx_s, m_s in enumerate(matches):
                             if idx_s == i: continue # Skip self
                             if m_s['match_text'] == match_txt:
                                 # Potential candidate
                                 # We re-use logic: 'adj_text' in m_s['full_text'] robustly?
                                 # FIX: Only apply if the replacement text actually makes sense there.
                                 # BUT the user might intentionally be REPLACING wrong text.
                                 similar_indices.append(idx_s)
                        
                         if similar_indices:
                             # Warning count
                             count_warn = len(similar_indices)
                             valid_similar_indices = []
                             
                             for idx_s in similar_indices:
                                 # STRICT CHECK: The new text MUST exist in the target sentence.
                                 target_sent_text = matches[idx_s]['full_text']
                                 
                                 # Find all occurrences of the NEW text
                                 import re
                                 # 1. Exact Match
                                 candidates_locs = [m for m in re.finditer(re.escape(adj_text), target_sent_text)]
                                 
                                 # 2. Relaxed if exact fails
                                 if not candidates_locs:
                                     # Try very loose match: ignore all spaces
                                     # Remove spaces from pattern
                                     pat_nospace = re.escape(adj_text.replace(" ", ""))
                                     # Insert \s* between every char? No, that's too much.
                                     # Just replace the original spaces with \s*
                                     pat_lax = re.escape(adj_text).replace(r"\ ", r"\s*") 
                                     candidates_locs = [m for m in re.finditer(pat_lax, target_sent_text, re.IGNORECASE)]

                                 is_valid_loc = False
                                 # Get original match location in the target sentence
                                 # Note: matches[idx_s] has start/end relative to its full_text
                                 orig_start = matches[idx_s]['start'] 
                                 orig_end = matches[idx_s]['end']
                                 
                                 for m_cand in candidates_locs:
                                     cand_start = m_cand.start()
                                     cand_end = m_cand.end()
                                     
                                     # Distance Logic
                                     # If cand starts after orig ends: cand_start - orig_end
                                     # If orig starts after cand ends: orig_start - cand_end
                                     # Overlap: negative
                                     dist = max(cand_start - orig_end, orig_start - cand_end)
                                     
                                     # Check for overlap or proximity (200 chars)
                                     if dist < 200:
                                         is_valid_loc = True
                                         break
                                 
                                 if is_valid_loc: 
                                     valid_similar_indices.append(idx_s)
                                 
                                 # FALLBACK: Explicitly handle the user's case where they select a completely different adjacent span.
                                 elif len(candidates_locs) == 1 and len(adj_text) > 4: # Relaxed length check mostly useful for years or short codes
                                     valid_similar_indices.append(idx_s)

                             if valid_similar_indices:
                                 # Initialize payload here to avoid scope issues if loop below tries to append
                                 # Actually, the button logic needs to be careful.
                                 # If the button is clicked, we need batch_payload.
                                 
                                 clicked_sim = st.form_submit_button(f"⚡ Apply to {len(valid_similar_indices)} Matches", help=f"Found {len(valid_similar_indices)} other sentences containing exactly '{adj_text}'.", key=f"btn_apply_sim_{u_key}")
                                 
                                 if clicked_sim:
                                     # Apply logic
                                     batch_payload = []
                                 
                                     # USER REQUEST: When applying to similar, ALWAYS apply the label from the source editor.
                                     # Even if I only changed boundaries, if I say this is GPE, the similar ones should also be GPE
                                     # (previously they might have been ORG or invalid).
                                     target_lbl = new_lbl 

                                     for idx_s in valid_similar_indices: # Use VALID indices, not all similar
                                         m_s = matches[idx_s]
                                         # Reconstruct key using stable format (no index 'i')
                                         # We need to determine 'current_label' logic for remote item
                                         
                                         stat_s = m_s.get('status', 'new')
                                         ov_lbl_s = m_s.get('overlap_label', '')
                                         if stat_s in ['accepted', 'pending'] and ov_lbl_s:
                                             curr_lbl_s = ov_lbl_s
                                         else:
                                             curr_lbl_s = m_s.get('label', label_candidate)
                                             
                                         uk_s = f"{m_s['sentence_id']}_{m_s['start']}_{curr_lbl_s}"
                                         
                                         # Prepare Update Dict
                                         update_item = {
                                             f"txt_{uk_s}": adj_text,
                                             f"lbl_{uk_s}": target_lbl,
                                             f"c_{uk_s}": True
                                         }
                                         batch_payload.append(update_item)
                                     
                                     # Store in Session State for Applying Next Run
                                     st.session_state.pending_batch_update = batch_payload
                                     
                                     st.toast(f"Scheduling update for {len(valid_similar_indices)} items...", icon="⚡")
                                     st.rerun()

                     # --- TEXT TRANSFER (PREV / CURR / NEXT) ---
                     # New Feature: Granular Text Moving via Database Tools
                     with st.expander("✂️ Edit Sentence Text (Transfer / Move)", expanded=False):
                         st.info("Automatically pull words/tokens from adjacent sentences in the document. Annotations will safely move along with the text!")

                         # Get exact neighbors safely via API
                         sid = m['sentence_id']
                         curr_src_doc_row = db.conn.execute("SELECT source_doc FROM sentences WHERE id = ?", (sid,)).fetchone()
                         curr_src_doc = curr_src_doc_row[0] if curr_src_doc_row else None
                         try:
                             adj = db.get_adjacent_sentences(sid, curr_src_doc)
                             
                             col_pull1, col_pull2, col_pull3 = st.columns([1.2, 1.5, 1.2])
                             
                             with col_pull1:
                                 if adj.get('prev'):
                                     if st.form_submit_button("⬅️ Pull Word from Previous", use_container_width=True, key=f"pull_prev_{u_key}"):
                                         success = db.pull_word_from_previous_sentence(sid, curr_src_doc)
                                         if success:
                                             st.rerun()
                                     st.caption(f"**Prev:** {adj['prev']['text']}")
                                 else:
                                     st.caption("No previous sentence in this document.")
                                     
                             with col_pull2:
                                 st.markdown(f"**Current Text:**\n\n> {m['full_text']}")
                                 
                             with col_pull3:
                                 if adj.get('next'):
                                     if st.form_submit_button("Pull Word from Next ➡️", use_container_width=True, key=f"pull_next_{u_key}"):
                                         success = db.pull_word_from_next_sentence(sid, curr_src_doc)
                                         if success:
                                             st.rerun()
                                     st.caption(f"**Next:** {adj['next']['text']}")
                                 else:
                                       st.caption("No next sentence in this document.")
                                       
                         except Exception as e:
                             st.error(f"Error loading adjacent sentences: {e}")

                     # --- TRASH BUTTON (Directly Delete existing annotations) ---
                     # Ensure we have an ID to delete and it's not a 'new' proposal
                     if m.get('overlap_id') and m.get('status') in ['accepted', 'pending']:
                         # Use a distinct styling for removal
                         if st.form_submit_button("🗑️ Reject (Delete)", help="Mark this annotation as Incorrect/Rejected.", key=f"del_ann_{u_key}"):
                             try:
                                 # Execute Deletion (Mark Rejected) immediately
                                 db.conn.execute("""
                                    UPDATE annotations 
                                    SET is_rejected = 1, 
                                        is_accepted = 0, 
                                        action_history = CASE 
                                            WHEN action_history IS NULL OR action_history = '' THEN 'manual_reject' 
                                            ELSE action_history || ',manual_reject' 
                                        END 
                                    WHERE id = ?
                                 """, (m['overlap_id'],))
                                 db.conn.commit()
                                 st.toast("Annotation Rejected!", icon="🗑️")
                                 time.sleep(0.5)
                                 st.rerun()
                             except Exception as e:
                                 st.error(f"Deletion failed: {e}")

                     # --- AUTO-SELECT LOGIC ---
                     # If user modified Label or Text, consider it selected
                     user_mod_lbl = (new_lbl != current_label)
                     user_mod_text = (adj_text != match_txt)
                     
                     if is_checked or user_mod_lbl or user_mod_text:
                         if i not in selected_indices:
                             selected_indices.append(i)

             # Submit Button
             st.info("💡 Tip: You can edit multiple items above. Changes are applied ONLY when you click the button below. Do not press Enter inside the text boxes.")
             
             main_submit = st.form_submit_button(f"✅ Process Page ({len(page_matches)} Items)")
             if main_submit:
                 # --- 1. EXECUTE PENDING SENTENCE MERGES ---
                 # We must do this BEFORE adding annotations, because the sentence IDs might change (deletion).
                 id_remap = {} # old_id -> (new_id, offset_added)
                 
                 if "pending_merges" in st.session_state and st.session_state.pending_merges:
                     pm_ids = sorted(list(st.session_state.pending_merges))
                     with sqlite3.connect(db_path) as conn:
                         cursor = conn.cursor()
                         count_merged = 0
                         
                         # Chained Merge Logic
                         # We maintain a mapping of where each ID has moved. 
                         # initially: id -> id
                         # When merging A -> B (B into A):
                         #  text(target(A)) += text(target(B))
                         #  target(B) is now dead.
                         #  remap[target(B)] = target(A) and all who pointed to t(B) now point to t(A).
                         
                         effective_id_map = {} # original_id -> current_valid_id
                         effective_offset_map = {} # original_id -> offset in current_valid_id
                         
                         def get_eff_id(iid):
                             if iid not in effective_id_map:
                                 return iid
                             # Path compression loop
                             curr = iid
                             while curr in effective_id_map and effective_id_map[curr] != curr:
                                 curr = effective_id_map[curr]
                             return curr
                         
                         # We iterate ordered by ID 
                         for psid in pm_ids:
                             try:
                                 # We want to merge psid AND psid+1
                                 # But psid might have been merged into (psid-1) already.
                                 # And psid+1 might be a fresh sentence.
                                 
                                 left_id = get_eff_id(psid)
                                 right_id = get_eff_id(psid + 1)
                                 
                                 if left_id == right_id:
                                     continue # Already merged?
                                     
                                 # Fetch Texts
                                 # Note: The DB text for left_id might have ALREADY been updated in this loop!
                                 # So we read from DB which holds the intermediate state.
                                 
                                 l_row = cursor.execute("SELECT text FROM sentences WHERE id = ?", (left_id,)).fetchone()
                                 r_row = cursor.execute("SELECT text FROM sentences WHERE id = ?", (right_id,)).fetchone()
                                 
                                 if not l_row or not r_row:
                                     continue
                                     
                                 txt_left = l_row[0]
                                 txt_right = r_row[0]
                                 
                                 offset = len(txt_left) + 1
                                 
                                 # Perform Merge in DB
                                 new_full = txt_left + " " + txt_right
                                 cursor.execute("UPDATE sentences SET text = ? WHERE id = ?", (new_full, left_id))
                                 
                                 # Move annotations
                                 # We update start_char/end_char AND append 'moved_merged' to history
                                 cursor.execute("""
                                    UPDATE annotations 
                                    SET sentence_id = ?, 
                                        start_char = start_char + ?, 
                                        end_char = end_char + ?,
                                        action_history = CASE 
                                            WHEN action_history IS NULL OR action_history = '' THEN 'moved_merged' 
                                            ELSE action_history || ',moved_merged' 
                                        END
                                    WHERE sentence_id = ?
                                 """, (left_id, offset, offset, right_id))
                                 
                                 # Delete Right
                                 cursor.execute("DELETE FROM sentences WHERE id = ?", (right_id,))
                                 if cursor.rowcount == 0:
                                     st.warning(f"Could not delete sentence {right_id}, possibly already deleted.")
                                 else:
                                     # Log internally if needed, logic is implicit success
                                     pass
                                 
                                 # Update Chains
                                 # Right ID acts as if it is now Left ID (plus offset)
                                 # We must remember this for the annotations we process later in the page
                                 # AND for the loop itself
                                 
                                 # For the loop "pointers":
                                 effective_id_map[right_id] = left_id
                                 # Note: Any future reference to right_id (e.g. merging right_id with right_id+1) 
                                 # will now resolve to left_id via get_eff_id.
                                 
                                 # For the final "id_remap" used by page_matches loop
                                 # We need to trace cumulative offsets. 
                                 # For simple neighbor chain: A+B+C. 
                                 # 1. A+B. B->A (off=lenA). 
                                 # 2. (A+B)+C. C->A (off=len(A+B)). 
                                 
                                 # Store result for id_remap
                                 # If right_id was previously 11, now it is 10.
                                 # But wait, what if right_id was 12, effectively 11?
                                 # We need a robust global offset map? 
                                 # Simplified: We just record what happened here.
                                 # Our annotation correction loop below needs: "Where is original ID X now?"
                                 
                                 # If X was mapped to Y with off1. And now Y merges into Z with off2.
                                 # Then X maps to Z with off1 + off2.
                                 
                                 # Update ALL keys in id_remap that point to right_id
                                 for k in list(id_remap.keys()):
                                     (curr_target, curr_off) = id_remap[k]
                                     if curr_target == right_id:
                                         id_remap[k] = (left_id, curr_off + offset)
                                         
                                 # Add right_id itself
                                 id_remap[right_id] = (left_id, offset)
                                 
                                 count_merged += 1
                                 
                             except Exception as e:
                                 st.error(f"Merge failed for ID {psid}: {e}")
                                 
                         conn.commit()
                         if count_merged > 0:
                             st.toast(f"Merged {count_merged} sentences!", icon="🔀")
                             st.session_state.pending_merges.clear()

                 count_new = 0
                 count_upd = 0
                 bar = st.progress(0)
                 
                 # Process only the selected indices that were on this page
                 # Note: selected_indices contains global 'i' indices
                 current_page_indices = [ix for ix in selected_indices if start_idx <= ix < end_idx]
                 
                 for idx_progress, idx in enumerate(current_page_indices):
                     m = matches[idx]
                     
                     # RECONSTRUCT KEY EXACTLY AS IN DISPLAY LOOP
                     # Logic must match lines ~760
                     stat_k = m.get('status', 'new')
                     ov_lbl_k = m.get('overlap_label', '')
                     if stat_k in ['accepted', 'pending'] and ov_lbl_k:
                         curr_lbl_k = ov_lbl_k
                     else:
                         curr_lbl_k = m.get('label', label_candidate)
                     
                     u_key = f"{m['sentence_id']}_{m['start']}_{curr_lbl_k}"
                     
                     # Get User Inputs (using Unique Key)
                     chosen_label = st.session_state.get(f"lbl_{u_key}", curr_lbl_k)
                     chosen_text = st.session_state.get(f"txt_{u_key}", m['match_text'])
                     
                     # Resolve Sentence ID and Offsets (Handle Merges)
                     s_id_final = m['sentence_id']
                     start_final = m['start']
                     end_final = m['end']
                     
                     if s_id_final in id_remap:
                         new_sid, off = id_remap[s_id_final]
                         s_id_final = new_sid
                         start_final += off
                         end_final += off
                     
# Default to original position/text if no valid change found
                     fin_start = start_final
                     fin_end = end_final
                     fin_text = m['match_text']

                     # Only attempt to find new coordinates if the user actually changed the text
                     if chosen_text != m['match_text']:
                         import re
                         
                         # Refresh context
                         try:
                            row_txt = st.session_state.db.conn.execute("SELECT text FROM sentences WHERE id = ?", (s_id_final,)).fetchone()
                            scan_ctx = row_txt[0] if row_txt else m['full_text']
                         except:
                            scan_ctx = m['full_text']
                        
                         # STRICT CHECK: Text must exist
                         if chosen_text not in scan_ctx:
                              st.error(f"❌ SKIPPED {m.get('id', 'N/A')}: The expanded text was not found in this sentence.")
                              continue

                         # If text exists, find the occurrence closest to the original position
                         pat = re.escape(chosen_text).replace(r"\ ", r"\s*") # Allow optional spaces
                         candidates = list(re.finditer(pat, scan_ctx, re.IGNORECASE))
                         
                         if candidates:
                             # Use strict overlap check
                             orig_rng = set(range(start_final, end_final))
                             valid_candidate = None
                             
                             # Sort candidates by distance to original start
                             candidates.sort(key=lambda x: abs(x.start() - start_final))
                             
                             # RELAXED VALIDATION LOGIC
                             if len(candidates) == 1 and len(chosen_text) > 4:
                                 # If unique and substantial, trust it completely
                                 valid_candidate = candidates[0]
                             else:
                                 for cand in candidates:
                                     cand_dist = abs(cand.start() - start_final)
                                     cand_rng = set(range(cand.start(), cand.end()))
                                     
                                     # PRIMARY CONDITION: Must overlap with original match
                                     if orig_rng.intersection(cand_rng):
                                         valid_candidate = cand
                                         break
                                     
                                     # SECONDARY CONDITION: Proximity check
                                     # Allow if within 200 chars distance
                                     if cand_dist < 200:
                                          # Use this candidate if it's the closest one
                                          valid_candidate = cand
                                          break
                                      
                             if not valid_candidate:
                                  st.error(f"❌ SKIPPED {m.get('id', 'N/A')}: New text found but not overlapping original position.")
                                  continue
                             
                             fin_start = valid_candidate.start()
                             fin_end = valid_candidate.end()
                             fin_text = scan_ctx[fin_start:fin_end]
                             
                         else:
                             st.error(f"Skipping {m['id']}: Text search failed.")
                             continue
                     
                     # Determine Action based on status
                     current_success = True
                     
                     # Calculate change types
                     changes = []
                     if chosen_text != m['match_text']:
                         changes.append("boundary_fixed")
                     if chosen_label != m.get('label', label_candidate) and chosen_label != m.get('overlap_label'):
                         changes.append("label_fixed")
                     
                     action_note = ",".join(changes)

                     if m.get('overlap_id'):
                         # UPDATE EXISTING (Accepted OR Pending)
                         try:
                             # Fetch existing history first
                             curr_hist = db.conn.execute("SELECT action_history FROM annotations WHERE id = ?", (m['overlap_id'],)).fetchone()
                             curr_hist_str = curr_hist[0] if curr_hist and curr_hist[0] else ""
                             
                             # Append new actions if unique
                             new_hist_parts = [p for p in changes if p not in curr_hist_str.split(',')]
                             if new_hist_parts:
                                 if curr_hist_str:
                                     final_hist = curr_hist_str + "," + ",".join(new_hist_parts)
                                 else:
                                     final_hist = ",".join(new_hist_parts)
                             else:
                                 final_hist = curr_hist_str

                             # FORCE UPDATE anyway (even if no change detected here, maybe they just confirmed it)
                             # If user explicitly clicked Process, we treat it as a verification too.
                             if not final_hist and not changes:
                                 final_hist = "verified"

                             db.conn.execute("""
                                UPDATE annotations 
                                SET text_span = ?, label = ?, start_char = ?, end_char = ?, is_accepted = 1, is_rejected = 0, source_agent = 'manual_fix', action_history = ?
                                WHERE id = ?
                             """, (fin_text, chosen_label, fin_start, fin_end, final_hist, m['overlap_id']))
                             db.conn.commit()
                             count_upd += 1
                         except Exception as e:
                             st.error(f"Update failed: {e}")
                             current_success = False
                             
                     else:
                         # CREATE NEW
                         try:
                             initial_history = "created_manual"
                             if changes:
                                 initial_history += "," + ",".join(changes)

                             # Use method signature: sentence_id, label, start_offset, end_offset, text_content, source, confidence, is_accepted
                             # We need to manually execute INSERT to include action_history since helper method might not support it yet
                             # Or update helper `add_annotation`? 
                             # Let's just do RAW INSERT here for speed/customization
                             
                             cursor = db.conn.cursor()
                             cursor.execute("""
                                INSERT INTO annotations (sentence_id, label, start_char, end_char, text_span, source_agent, confidence, is_accepted, created_at, action_history)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                             """, (s_id_final, chosen_label, fin_start, fin_end, fin_text, 'manual_fix', 1.0, 1, initial_history))
                             db.conn.commit()
                             
                             count_new += 1
                         except Exception as e:
                             st.error(f"Creation failed: {e}")
                             current_success = False

                     if current_success:
                          # UPDATE SESSION STATE TO REFLECT "ACCEPTED" STATUS
                          # m is a reference to the item in matches/all_matches
                          m['status'] = 'accepted'
                          m['label'] = chosen_label
                          # If text changed, update it too for display consistency
                          m['match_text'] = fin_text 
                         
                 st.success(f"Processed: {count_new} new, {count_upd} updated.")
                 time.sleep(1)
                 st.rerun()
        
        # --- Bottom Pagination Controls ---
        if total_items > ITEMS_PER_PAGE:
            st.markdown("---")
            p1, p2, p3 = st.columns([1, 2, 1])
            with p1:
                # Need unique key so as not to conflict with previous
                if st.button("⬅️ Prev Page", key="con_prev_btm", disabled=(st.session_state.con_page == 0)):
                    st.session_state.con_page -= 1
                    st.rerun()
            with p2:
                # Re-calculate max page just in case
                max_page_b = max(0, (total_items - 1) // ITEMS_PER_PAGE)
                st.markdown(f"<div style='text-align:center;'>Page {st.session_state.con_page + 1} of {max_page_b + 1}</div>", unsafe_allow_html=True)
            with p3:
                if st.button("Next ➡️", key="con_next_btm", disabled=(st.session_state.con_page == max_page_b)):
                    st.session_state.con_page += 1
                    st.rerun()
            st.info("💡 Tip: Changes are lost if you change page without clicking 'Process' first.")

if "tab_remove" in locals():
    # --- TAB 2: REMOVE / FIX ANNOTATIONS ---
    with tab_remove:
        st.subheader("2. Remove or Fix Annotations")
        st.info("Find existing annotations (Pending or Accepted) and remove/fix them using detection patterns.")
        
        # --- Control Panel ---
        
        # 0. Global List Management (Shared with Tab 1)
        # We reuse st.session_state.vec_ents or .top_ents to avoid double-fetching
        source_list = []
        if "Vector" in st.session_state.get('rem_mode', '') or "Hybrid" in st.session_state.get('rem_mode', ''):
             if st.session_state.get("vec_ents"):
                 source_list = st.session_state.vec_ents
        else:
             if st.session_state.get("top_ents"):
                 source_list = st.session_state.top_ents
        
        # Fallback if empty (Tab 1 hasn't run yet)
        if not source_list:
             # Default to simple frequency list if nothing cached
             source_list = db.get_top_entities(limit=500)

        valid_rem_ents = [e for e in source_list if len(e['text']) >= 3]

        # 1. Detection Mode & Scope
        c1, c2 = st.columns([1, 1])
        with c1:
             st.markdown("**1. Detection Mode**")
             rem_mode = st.radio(
                "Mode ", 
                ["String Match (Exact)", "Fuzzy Match (Approximate)", "Vector Match (Semantic)", "Hybrid (Fuzzy + Vector)"], 
                horizontal=True,
                label_visibility="collapsed",
                key="rem_mode"
             )
        
        with c2:
             st.markdown("**2. Scope (Status)**")
             rem_scope = st.radio("Look in:", ["All Existing", "Pending (Suggestions)", "Accepted (Training Data)"], horizontal=True, index=0, key="rem_scope_radio")

        scope_map = {
            "All Existing": "all",
            "Pending (Suggestions)": "pending",
            "Accepted (Training Data)": "accepted"
        }

        st.divider()
        
        # 2. Target Selection (Entity List or Manual)
        col_sel_main, col_sel_opt = st.columns([3, 1])
        
        rem_selected_ent = None
        rem_manual_text = ""
        
        with col_sel_main:
             st.markdown("**3. Select Target Entity**")
             
             # Manual Toggle
             use_manual_rem = st.checkbox("✍️ Manual Search Input", key="rem_manual_chk", help="Type text manually instead of choosing from list.")
             
             if use_manual_rem:
                 rem_manual_text = st.text_input("Enter text to remove/fix:", placeholder="e.g. noise...", key="rem_man_input")
             else:
                 # Search Filter
                 rem_filter = st.text_input("🔍 Filter List:", placeholder="Start typing entity name...", key="rem_list_filter")
                 
                 # Dynamic Filtering
                 if rem_filter:
                    s_lower = rem_filter.lower()
                    filtered_rem = [e for e in valid_rem_ents if s_lower in e['text'].lower()]
                    filtered_rem.sort(key=lambda x: (0 if x['text'].lower() == s_lower else 1))
                 else:
                    filtered_rem = valid_rem_ents[:200]
                 
                 def get_fmt_rem(x): return f"{x['text']} [{x['label']}] ({x.get('count', '?')})"
                 
                 if filtered_rem:
                     rem_selected_ent = st.selectbox("Choose Entity:", filtered_rem, format_func=get_fmt_rem, key="rem_selectbox", label_visibility="collapsed")
                 else:
                     st.warning("No matches in list.")
        
        # Resolve 'rem_query' for the button logic below
        if use_manual_rem:
            rem_query = rem_manual_text
            current_rem_label = 'MANUAL'
        else:
            if rem_selected_ent:
                rem_query = rem_selected_ent['text']
                current_rem_label = rem_selected_ent['label']
            else:
                rem_query = ""
                current_rem_label = None

        if st.button("🔎 Scan for Removal", type="primary", disabled=(not rem_query), key="btn_rem_scan"):
            # We reuse the logic from Tab 1 but filter differently
            # For removal, we ONLY care about overlaps (where status != 'new')
            
            # 1. Perform Scan (Generic)
            # We need to adapt the scanner to return matches
            # The existing scanners (String/Fuzzy) return 'new' matches primarily but check for overlaps.
            # We need them to return the overlaps as the primary result.
            
            with st.spinner("Scanning annotated corpus..."):
                found_annotations = []
                
                # A. String Match
                if rem_mode == "String Match (Exact)":
                     # Direct DB Query for Annotations containing the string
                     status_sql = ""
                     if rem_scope == "Pending (Suggestions)": status_sql = "AND is_accepted = 0"
                     elif rem_scope == "Accepted (Training Data)": status_sql = "AND is_accepted = 1"
                     
                     search_pattern = f"%{rem_query}%"
                     c = db.conn.cursor()
                     # We use LIKE for substring match. Note: SQLite LIKE is case-insensitive for ASCII only by default.
                     # This query finds annotations where your search text appears INSIDE the annotation text.
                     c.execute(f"""
                        SELECT id, text_span, label, sentence_id, is_accepted, start_char, end_char 
                        FROM annotations 
                        WHERE (is_rejected = 0 OR is_rejected IS NULL) 
                        AND text_span LIKE ? 
                        {status_sql}
                     """, (search_pattern,))
                     
                     rows = c.fetchall()
                     for (aid, text, label, sid, is_acc, s_char, e_char) in rows:
                         # Get sentence text for context
                         s_row = db.conn.execute("SELECT text FROM sentences WHERE id = ?", (sid,)).fetchone()
                         s_text = s_row[0] if s_row else "?"
                         
                         found_annotations.append({
                             'overlap_id': aid,
                             'match_text': text, # The FULL annotation text
                             'overlap_label': label,
                             'full_text': s_text,
                             'status': 'accepted' if is_acc else 'pending',
                             'score': 1.0, 
                             'start': s_char,
                             'end': e_char
                         })
                
                # B. Fuzzy Match
                elif rem_mode == "Fuzzy Match (Approximate)":
                     # We reuse the heavy fuzzy logic but we can optimize since we only care about
                     # checking existing annotations vs the query, not the whole text?
                     
                     status_sql = ""
                     if rem_scope == "Pending (Suggestions)": status_sql = "AND is_accepted = 0"
                     elif rem_scope == "Accepted (Training Data)": status_sql = "AND is_accepted = 1"
                     
                     c = db.conn.cursor()
                     # ADDED start_char, end_char to selection
                     c.execute(f"SELECT id, text_span, label, sentence_id, is_accepted, start_char, end_char FROM annotations WHERE (is_rejected = 0 OR is_rejected IS NULL) {status_sql}")
                     all_anns = c.fetchall()
                     
                     from rapidfuzz import fuzz
                     # Pre-computation
                     q_lower = rem_query.lower()
                     
                     for (aid, text, label, sid, is_acc, s_char, e_char) in all_anns:
                         t_lower = text.lower()
                         
                         # Shortcut: Exact match always passes (ignoring threshold)
                         if q_lower == t_lower:
                             ratio = 100.0
                         else:
                             ratio = fuzz.ratio(q_lower, t_lower)
                             
                         if ratio > 80: # Threshold
                             # Get sentence text for context
                             s_row = db.conn.execute("SELECT text FROM sentences WHERE id = ?", (sid,)).fetchone()
                             s_text = s_row[0] if s_row else "?"
                             
                             found_annotations.append({
                                 'overlap_id': aid,
                                 'match_text': text,
                                 'overlap_label': label,
                                 'full_text': s_text,
                                 'status': 'accepted' if is_acc else 'pending',
                                 'score': ratio,
                                 'start': s_char,
                                 'end': e_char
                             })
                
                # C. Vector Match
                elif rem_mode in ["Vector Match (Semantic)", "Hybrid (Fuzzy + Vector)"]:
                     vec = None
                     
                     # 1. Try Centroids first (Better)
                     # variable 'current_rem_label' comes from the logic above the button
                     if 'current_rem_label' in locals() and current_rem_label and current_rem_label != 'MANUAL':
                         c = db.conn.cursor()
                         c.execute("SELECT vector FROM entity_centroids WHERE text_span = ? AND label = ?", (rem_query, current_rem_label))
                         v_row = c.fetchone()
                         if v_row: vec = v_row[0]
                     
                     # 2. Fallback to Annotations
                     if not vec:
                         sql = "SELECT vector FROM annotations WHERE text_span = ? AND vector IS NOT NULL LIMIT 1"
                         args = (rem_query,)
                         # If we know the label, restrict it
                         if 'current_rem_label' in locals() and current_rem_label and current_rem_label != 'MANUAL':
                              sql = "SELECT vector FROM annotations WHERE text_span = ? AND label = ? AND vector IS NOT NULL LIMIT 1"
                              args = (rem_query, current_rem_label)
                              
                         v_row = db.conn.execute(sql, args).fetchone()
                         if v_row: vec = v_row[0]
                     
                     if not vec:
                         st.error(f"No vector found for '{rem_query}'. Cannot perform semantic search.")
                     else:
                        # Find similar annotations
                        sim_anns = db.get_similar_pending_annotations(
                            vec, 
                            threshold=0.75, 
                            limit=500, 
                            scope=scope_map[rem_scope]
                        )
                        # Convert to format
                        for sa in sim_anns:
                             found_annotations.append({
                                 'overlap_id': sa['id'],
                                 'match_text': sa['text_span'],
                                 'overlap_label': sa['label'],
                                 'full_text': sa['sentence_text'],
                                 'status': 'accepted' if sa.get('is_accepted') else 'pending',
                                 'score': sa['confidence'],
                                 'start': sa['start_char'],  # Added
                                 'end': sa['end_char']       # Added
                             })
                             
                        # Hybrid Filter
                        if "Hybrid" in rem_mode:
                            # Re-filter by fuzzy
                            found_annotations = [
                                x for x in found_annotations 
                                if fuzz.ratio(rem_query.lower(), x['match_text'].lower()) > 60
                            ]

                st.session_state.remove_matches = found_annotations
                st.rerun()

        # --- Display Removal Candidates ---
        if "remove_matches" in st.session_state and st.session_state.remove_matches:
            matches = st.session_state.remove_matches
            st.divider()
            
            # Header with Clear Button
            hm1, hm2 = st.columns([3, 1])
            hm1.subheader(f"Results: {len(matches)} Candidates for Removal")
            if hm2.button("🧹 Clear Results", key="btn_clear_rem"):
                 st.session_state.remove_matches = []
                 st.rerun()

            # Select All
            select_all_rem = st.checkbox("Select All", key="sel_all_rem")
            
            with st.form("remove_form"):
                
                # Pre-define colors (Matches Tab 1)
                TYPE_COLOR_MAP = {
                    "LEG-REFS": "#8A2BE2", "ORG": "#1E90FF", "PERSON": "#FF4500", "GPE": "#DAA520",
                    "DATE": "#2E8B57", "FACILITY": "#20B2AA", "LOCATION": "#CD853F", "PUBLIC_DOCS": "#708090"
                }

                to_remove_ids = []
                for i, m in enumerate(matches):
                     # Unique key: Include select_all interaction to force refresh if master toggle changes
                     # This ensures 'value' param is respected when select_all changes
                     u_key_rem = f"rem_item_{i}_{m.get('overlap_id', i)}_{select_all_rem}"
                     
                     col_chk, col_txt = st.columns([0.5, 10])
                     with col_chk:
                         # Fixed: label non-empty + collapsed visibility
                         is_checked = st.checkbox("Select", key=u_key_rem, value=select_all_rem, label_visibility="collapsed")
                         if is_checked:
                             to_remove_ids.append(m['overlap_id'])
                     
                     with col_txt:
                         # Render Badge Style (Simplified from Tab 1)
                         full_text = m.get('full_text', '')
                         match_txt = m.get('match_text', '?')
                         lbl = m.get('overlap_label', 'UNK')
                         status = m.get('status', '')
                         
                         trunc_color = TYPE_COLOR_MAP.get(lbl, "#555")
                         
                         # Status Badge
                         if status == 'accepted':
                             bs = f"<span style='background:#ccc; color:#555; font-size:0.75em; padding:2px 4px; border-radius:4px;'>🔒 Accepted</span>"
                             bg_row = "#f0f8ff"
                         else:
                             bs = f"<span style='background:#ffd700; color:#000; font-size:0.75em; padding:2px 4px; border-radius:4px;'>⚠️ Pending</span>"
                             bg_row = "#fffaf0"
                         
                         # Parse offsets if available (String/Vector modes have them now)
                         start = m.get('start', -1)
                         end = m.get('end', -1)
                         
                         if start >= 0 and end > start and len(full_text) > end:
                             ctx_s = max(0, start - 60)
                             ctx_e = min(len(full_text), end + 60)
                             
                             prefix = full_text[ctx_s:start]
                             suffix = full_text[end:ctx_e]
                             
                             inner_html = (
                                 f"<span style='background-color: rgba(0,0,0,0.03); border-bottom: 2px solid {trunc_color}; "
                                 f"padding: 0 3px; color: {trunc_color}; font-weight: bold; border-radius: 3px;'>"
                                 f"{match_txt}"
                                 f"<span style='background-color: {trunc_color}; color: white; margin-left: 6px; padding: 1px 5px; border-radius: 3px; font-size: 0.7em; vertical-align: middle;'>{lbl}</span>"
                                 f"</span>"
                             )
                             
                             html_snippet = f"<div style='font-family: sans-serif; line-height: 1.5; background: {bg_row}; padding: 8px; border-radius: 5px; border: 1px solid #eee; font-size: 0.9em;'>" \
                                            f"{bs} ...{prefix}{inner_html}{suffix}...</div>"
                         else:
                             # Fallback if offsets missing or invalid
                             html_snippet = f"<div style='background: {bg_row}; padding: 8px; border-radius: 5px;'>{bs} <b>{match_txt}</b> ({lbl}) <br><small>...{full_text[:150]}...</small></div>"

                         st.markdown(html_snippet, unsafe_allow_html=True)
                
                # Correct Logic for Form Submit
                if st.form_submit_button("🗑️ Delete Selected Annotations"):
                    if to_remove_ids:
                        db.delete_annotations(to_remove_ids)
                        st.success(f"Deleted {len(to_remove_ids)} annotations.")
                        time.sleep(1)
                        # Clear state
                        st.session_state.remove_matches = []
                        st.rerun()
                    else:
                        st.warning("No selections made.")


with tab_high_conf:
    st.subheader("3. High Confidence > 90% (False Negatives)")
    st.markdown("Entities the model is **sure** about, but you missed.")
    
    if st.button("🚀 Find High Confidence Suggestions"):
        try:
           # We need predictions stored in DB as candidates first
           # For now, let's look for pending annotations if they exist in DB
            candidates = db.get_high_confidence_candidates(min_conf=0.90, limit=50)
            if candidates:
                st.session_state.hi_conf_candidates = candidates
            else:
                st.info("No high confidence candidates found in DB. Run the model prediction pipeline first.")
        except AttributeError:
             st.error("DBManager method missing. Updating...")

    if "hi_conf_candidates" in st.session_state:
        cands = st.session_state.hi_conf_candidates
        st.write(f"Found {len(cands)} candidates.")
        
        with st.form("hi_conf_batch"):
            # Selectable dataframe
            selections = []
            for i, c in enumerate(cands):
                chk = st.checkbox(f"Accept: **{c['text']}** ({c['label']}) in: _{c['context']}_", value=True, key=f"hi_{i}")
                if chk:
                    selections.append(c)
            
            if st.form_submit_button(f"✅ Batch Accept Selected ({len(selections)})"):
                progress = st.progress(0)
                for i, s in enumerate(selections):
                    db.accept_annotation(s['id'])
                    progress.progress((i+1)/len(selections))
                st.success(f"Accepted {len(selections)} suggestions!")
                time.sleep(1)
                st.session_state.hi_conf_candidates = [] # Clear cache
                st.rerun()


if "tab_repair" in locals():
    with tab_repair:
        st.subheader("4. Repair Fragmented Sentences")
        st.info("Detect and merge sentences that were incorrectly split (Splits inside Entities, Title separations, or Abbreviations).")

        # MODE SELECTOR
        rep_mode = st.radio("Review Mode:", ["👀 Manual Sequential Review", "🔎 Scanner (Heuristic Batch)"], horizontal=True, label_visibility="visible")

        # --- MODE 1: MANUAL REVIEW ---
        if rep_mode == "👀 Manual Sequential Review":
            # 1. Initialize State
            if "rep_cursor" not in st.session_state:
                # Get min ID
                min_row = db.conn.execute("SELECT MIN(id) FROM sentences").fetchone()
                st.session_state.rep_cursor = min_row[0] if min_row and min_row[0] else 1

            # 2. Controls & Navigation
            # Use columns for layout
            cn1, cn2, cn3, cn4 = st.columns([1, 1, 2, 2])
            
            with cn1:
                if st.button("⬅️ PREV", key="btn_prev", use_container_width=True):
                    prev_row = db.conn.execute("SELECT id FROM sentences WHERE id < ? ORDER BY id DESC LIMIT 1", (st.session_state.rep_cursor,)).fetchone()
                    if prev_row: 
                        st.session_state.rep_cursor = prev_row[0]
                        st.rerun()

            with cn2:
                if st.button("NEXT ➡️", key="btn_next", use_container_width=True):
                    next_row = db.conn.execute("SELECT id FROM sentences WHERE id > ? ORDER BY id ASC LIMIT 1", (st.session_state.rep_cursor,)).fetchone()
                    if next_row: 
                        st.session_state.rep_cursor = next_row[0]
                        st.rerun()
            
            with cn3:
                # Direct Jump (with dynamic key to ensure update)
                curs = st.session_state.rep_cursor
                def on_idx_change():
                    st.session_state.rep_cursor = st.session_state[f"goto_{curs}"]
                
                # Trick: Using dynamic key forces re-render if cursor changes externally (via buttons)
                st.number_input("Go to ID", value=curs, key=f"goto_{curs}", on_change=on_idx_change, label_visibility="collapsed")

            # 3. Fetch Data (Current & Next)
            curr_id = st.session_state.rep_cursor
            row_curr = db.conn.execute("SELECT id, text, source_doc FROM sentences WHERE id = ?", (curr_id,)).fetchone()
            
            if not row_curr:
                st.warning(f"Sentence {curr_id} not found.")
            else:
                current_text_db = row_curr[1]
                curr_src_doc = row_curr[2]
                row_next = db.conn.execute("SELECT id, text FROM sentences WHERE id > ? AND source_doc = ? ORDER BY id ASC LIMIT 1", (curr_id, curr_src_doc)).fetchone()
                
                # Layout
                c_lbl1, c_lbl2 = st.columns([1, 1])
                c_lbl1.caption(f"🔵 **Current Sentence** (ID: {curr_id})")
                if row_next: c_lbl2.caption(f"⚪ **Next Sentence** (ID: {row_next[0]})")
                
                c_edit, c_view = st.columns([1, 1])
                
                with c_edit:
                    # EDITABLE Current Sentence (fixes "Grey" issue + allows Typo fix)
                    # We use session state to track edits
                    # CRITICAL: Key must be dynamic per ID, otherwise Streamlit holds stale state from previous sentences
                    txt_curr_val = st.text_area("Current Text", value=current_text_db, height=200, label_visibility="collapsed", key=f"txt_curr_{curr_id}")
                
                with c_view:
                    if row_next:
                        next_id = row_next[0]
                        next_text_db = row_next[1]
                        st.text_area("Next Text", value=next_text_db, height=200, disabled=False, label_visibility="collapsed", key=f"txt_next_{next_id}") # Disabled=False but ignored in logic
                    else:
                        st.info("No next sentence (End of DB).")

                # ACTIONS
                c_act1, c_act2, c_act3, c_act4 = st.columns([1, 1.5, 1.5, 1.5])
                with c_act1:
                     # SAVE CHANGES TO CURRENT (If edited)
                     if txt_curr_val != current_text_db:
                         if st.button("💾 Save Text", use_container_width=True, key=f"save_{curr_id}"):
                             db.conn.execute("UPDATE sentences SET text = ? WHERE id = ?", (txt_curr_val, curr_id))
                             db.conn.commit()
                             st.toast("Updated text!")
                             st.rerun()

                with c_act2:
                    if st.button("⬅️ Pull Word from Prev", use_container_width=True, key=f"pull_p_{curr_id}"):
                        if db.pull_word_from_previous_sentence(curr_id, curr_src_doc): st.rerun()
                        else: st.warning("Cannot pull from previous.")
                
                with c_act3:
                    if st.button("Pull Word from Next ➡️", use_container_width=True, key=f"pull_n_{curr_id}"):
                        if db.pull_word_from_next_sentence(curr_id, curr_src_doc): st.rerun()
                        else: st.warning("Cannot pull from next.")

                with c_act4:
                    if row_next:
                        if st.button("🔗 MERGE ALL NEXT", type="primary", use_container_width=True, key=f"merge_{curr_id}_{next_id}", help="Legacy: Consume ENTIRE next sentence"):
                             # MERGE LOGIC (Using text from WIDGET to capture edits)
                             id1 = curr_id
                             txt1 = txt_curr_val # USE USER EDITED VERSION
                             id2 = row_next[0]
                             txt2 = row_next[1]
                             
                             new_text = txt1.strip() + " " + txt2.strip()
                             offset_shift = len(txt1.strip()) + 1
                             
                             try:
                                 # 1. Update S1
                                 db.conn.execute("UPDATE sentences SET text = ? WHERE id = ?", (new_text, id1))
                                 # 2. Move Anns
                                 anns_s2 = db.conn.execute("SELECT id, start_char, end_char FROM annotations WHERE sentence_id = ?", (id2,)).fetchall()
                                 for (aid, s, e) in anns_s2:
                                     new_s = s + offset_shift
                                     new_e = e + offset_shift
                                     db.conn.execute("UPDATE annotations SET sentence_id = ?, start_char = ?, end_char = ? WHERE id = ?", (id1, new_s, new_e, aid))
                                 # 3. Delete S2
                                 db.conn.execute("DELETE FROM sentences WHERE id = ?", (id2,))
                                 db.conn.commit()
                                 st.toast(f"Merged {id1} + {id2}!")
                                 time.sleep(0.5)
                                 # Do NOT advance cursor automatically (User often needs to chain-merge S1+S2+S3...)
                                 st.rerun() 
                             except Exception as e:
                                 st.error(f"Error: {e}")
                        
                # Hints
                if row_next:
                     # Check if join looks good
                     join_hint = ""
                     c_txt = row_curr[1].strip()
                     n_txt = row_next[1].strip()
                     
                     abbrs = ["Αριθ.", "αριθ.", "αρ.", "τ.κ.", "Τ.Κ.", "κ.", "π.δ.", "ν.", "δηλ.", "κ.λπ.", "χλμ.", "εκ.", "χιλ."]
                     if len(c_txt) < 4: join_hint = "Short sentence"
                     elif any(c_txt.endswith(a) for a in abbrs): join_hint = "Ends with Abbreviation"
                     elif n_txt[0].islower(): join_hint = "Next starts lowercase"
                     
                     if join_hint:
                         st.success(f"💡 Suggestion: Merge available ({join_hint})")

        # --- MODE 2: SCANNER (Heuristic) ---
        elif rep_mode == "🔎 Scanner (Heuristic Batch)":
            # Configuration for Scan
            c_cfg1, c_cfg2 = st.columns(2)
            with c_cfg1:
                use_lexicon = st.checkbox("🧩 Use Existing Entities as Dictionary", value=True, help="Checks if joining the sentences creates a known entity name found elsewhere in the DB.")
            with c_cfg2:
                check_titles = st.checkbox("📜 Check for Detached Titles", value=True, help="Checks if a Legal Reference is split from its Title (e.g. 'Law 4000/2010. Regarding...')")

            if st.button("🔎 Scan for Splits"):
                with st.spinner("Analyzing sentence boundaries & checking Knowledge Base..."):
                    try:
                        candidates = []
                        cursor = db.conn.cursor()
                        
                        # 1. PREP: Get Lexicon (if enabled)
                        # We grab distinct entity texts longer than 15 chars (shorter ones cause false positives)
                        lexicon = set()
                        if use_lexicon:
                            cursor.execute("SELECT DISTINCT text_span FROM annotations WHERE length(text_span) > 15 LIMIT 5000")
                            lexicon = {row[0] for row in cursor.fetchall()}
                        
                        # 2. PREP: Fetch Sentences
                        # We limit to 50k for performance, or user filter could apply in future
                        cursor.execute("SELECT id, text FROM sentences ORDER BY id")
                        all_sents = cursor.fetchall()
                        
                        # 3. HELPER PATTERNS
                        abbrs = ["Αριθ.", "αριθ.", "αρ.", "τ.κ.", "Τ.Κ.", "κ.", "π.δ.", "ν.", "δηλ.", "κ.λπ.", "χλμ.", "εκ.", "χιλ."]
                        title_starters = ["Περί", "Για", "Κύρωση", "Τροποποίηση", "Αντικατάσταση", "Ενσωμάτωση", "ΦΕΚ", "Με θέμα"]
                        
                        # 4. SCAN LOOP
                        for i in range(len(all_sents) - 1):
                            cid, ctext = all_sents[i]
                            nid, ntext = all_sents[i+1]
                            
                            ctext_strip = ctext.strip()
                            ntext_strip = ntext.strip()
                            if not ctext_strip or not ntext_strip: continue

                            reason = None
                            
                            # --- CHECK A: Abbreviation Split (Classic) ---
                            if len(ctext_strip) < 3: 
                                reason = "Very short sentence"
                            elif any(ctext_strip.endswith(a) for a in abbrs):
                                reason = "Ends with Abbreviation"
                            elif ntext_strip[0].islower():
                                reason = "Next starts with lowercase"
                                
                            # --- CHECK B: Split Pattern in Title (User Request) ---
                            # Logic: Sentence 1 looks like "N. 1234/2000" and Sentence 2 looks like "About something"
                            if check_titles:
                                # Heuristic: Ends with digit, next starts with Title Keyword or Capital
                                # e.g. "N. 4172/2013." -> "Fek A 167."
                                if ctext_strip[-1].isdigit() or ctext_strip.endswith(")"): 
                                    if any(ntext_strip.startswith(t) for t in title_starters):
                                        reason = f"Possible Detached Title ('{ntext_strip.split()[0]}...')"
                            
                            # --- CHECK C: Entity Re-assembly (Lexicon) ---
                            if use_lexicon and not reason:
                                # Try joining with space
                                joined = ctext_strip + " " + ntext_strip
                                # Check if the boundary allows for a known entity
                                # Optimization: Check if any known entity is in the JOINED string but NOT in the components individually
                                # (This is expensive, so we simplify: does the junction match a known entity?)
                                
                                # Simple approach: Check if joined text contains a known entity that bridges the gap
                                # We take last 20 chars of S1 and first 20 of S2
                                bridge = ctext_strip[-25:] + " " + ntext_strip[:25]
                                
                                # This is O(N*M) heavy. Better heuristic:
                                # string match specific long entities? 
                                # Let's search if the 'bridge' exists in lexicon.
                                # Actually, Python's `in` is fast.
                                # Let's verify against lexicon items that COULD match.
                                # Just check if 'bridge' is IN any lexicon item is too slow (5000 items).
                                
                                # Reverse: Check if any lexicon item is in bridge.
                                for lex in lexicon:
                                    if lex in bridge:
                                        # Crucial: verify the entity actually CROSSES the split
                                        # i.e., it's not just fully inside S1 or S2
                                        part1 = lex.split()[0]
                                        part2 = lex.split()[-1]
                                        if part1 in ctext_strip and part2 in ntext_strip:
                                            reason = f"Reconstructs Known Entity: '{lex[:30]}...'"
                                            break
                            
                            if reason:
                                 candidates.append({
                                     'id_1': cid, 'text_1': ctext,
                                     'id_2': nid, 'text_2': ntext,
                                     'reason': reason
                                 })
                                 
                            if len(candidates) >= 100: break # Cap results

                        st.session_state.repair_candidates = candidates
                    except Exception as e:
                        st.error(f"Scan failed: {e}")

        if "repair_candidates" in st.session_state and st.session_state.repair_candidates and rep_mode == "🔎 Scanner (Heuristic Batch)":
             cands = st.session_state.repair_candidates
             
             # Filter/Sort options
             filter_reason = st.selectbox("Filter by Problem Type:", ["All"] + list(set(c['reason'] for c in cands)))
             filtered_cands = [c for c in cands if filter_reason == "All" or c['reason'] == filter_reason]
             
             st.write(f"Showing {len(filtered_cands)} / {len(cands)} candidates.")
             
             to_merge = []
             
             with st.form("repair_form"):
                 for i, c in enumerate(filtered_cands):
                     st.markdown(f"**#{i+1}** <span style='color:orange'>({c['reason']})</span>", unsafe_allow_html=True)
                     c1, c2, c3 = st.columns([0.5, 4, 4])
                     
                     # Checkbox
                     is_chk = c1.checkbox("Merge", key=f"mrg_{c['id_1']}")
                     
                     # Visual
                     with c2: 
                         st.caption(f"Sentence {c['id_1']}")
                         match_color = "background-color: #ffcccc;" if "Entity" in c['reason'] else ""
                         st.markdown(f"<div style='{match_color} padding:5px; border-radius:5px'>{c['text_1']}</div>", unsafe_allow_html=True)
                     with c3:
                         st.caption(f"Sentence {c['id_2']}")
                         st.info(c['text_2'])
                 
                 if is_chk:
                     to_merge.append(c)
                 st.divider()

             if st.form_submit_button("🔗 Merge Selected Sentences"):
                 if not to_merge:
                     st.warning("Select sentences to merge.")
                 else:
                     count = 0
                     prog = st.progress(0)
                     
                     for idx_m, item in enumerate(to_merge):
                         # PERFORM MERGE
                         id1 = item['id_1']
                         id2 = item['id_2']
                         txt1 = item['text_1']
                         txt2 = item['text_2']
                         
                         new_text = txt1.strip() + " " + txt2.strip()
                         offset_shift = len(txt1.strip()) + 1
                         
                         try:
                             # 1. Update Sentence 1
                             db.conn.execute("UPDATE sentences SET text = ? WHERE id = ?", (new_text, id1))
                             
                             # 2. Move Annotations from S2 -> S1 (and shift offsets)
                             # We need to get them first to calculate new offsets
                             anns_s2 = db.conn.execute("SELECT id, start_char, end_char FROM annotations WHERE sentence_id = ?", (id2,)).fetchall()
                             
                             for (aid, s, e) in anns_s2:
                                 new_s = s + offset_shift
                                 new_e = e + offset_shift
                                 db.conn.execute("UPDATE annotations SET sentence_id = ?, start_char = ?, end_char = ? WHERE id = ?", (id1, new_s, new_e, aid))
                             
                             # 3. Delete Sentence 2
                             db.conn.execute("DELETE FROM sentences WHERE id = ?", (id2,))
                             
                             db.conn.commit()
                             count += 1
                         except Exception as e:
                             st.error(f"Failed to merge {id1}+{id2}: {e}")
                             
                         prog.progress((idx_m + 1)/len(to_merge))
                     
                     st.success(f"Merged {count} sentence pairs successfully!")
                     st.session_state.repair_candidates = []
                     time.sleep(1)
                     st.rerun()

if "tab_lexicon" in locals():
    with tab_lexicon:
        st.subheader("5. Knowledge Base Management")
        st.markdown("Edit the underlying **Lexicons (.json)** and **Regex Patterns (.txt)** that drive the rule-based pre-annotation, or **Rebuild Vector Memory**.")

        # --- MEMORY SECTION ---
        with st.expander("🧠 Vector Memory Management", expanded=True):
            active_db_name = os.path.basename(db.db_path)
            st.info(f"Rebuilds the 'Entity Centroids' (Vector Index) used for Semantic Search.\n\n📂 **Target DB:** `{active_db_name}`")

            if st.button("🧠 Build/Rebuild Memory Index"):
                try:
                    # Correct Imports based on file structure
                    from src.core.memory_manager import MemoryManager
                    from src.models.roberta_ner import RobertaNER

                    # Helper to load model (copied from Annotator.py logic)
                    def load_roberta_scan():
                        candidates = [
                            r"E:\ALGORITHMS\Semi-auto annotation model\to_move\src\agents\roberta_model\mdeberta-v3-base\seed_1",
                            "src/agents/GreekLegalRoBERTa_New",
                            "E:/ALGORITHMS/Semi-auto annotation model/src/agents/GreekLegalRoBERTa_New",
                            "src/agents/GreekLegalRoBERTa_v3",
                            "xlm-roberta-base"
                        ]
                        for path in candidates:
                             # Normalize check
                             if "xlm" in path or os.path.exists(path):
                                 try:
                                     return RobertaNER(path)
                                 except: continue
                        return RobertaNER("xlm-roberta-base")

                    with st.spinner("Initializing Neural Brain & Memory (This happens once)..."):
                        # 1. Ensure Model is Loaded
                        if "ner_model" not in st.session_state:
                             st.session_state.ner_model = load_roberta_scan()

                        ner_model = st.session_state.ner_model

                        # 2. Initialize Manager (Central Intelligence)
                        # Note: We pass the ACTIVE 'db' object and the loaded model
                        mm = MemoryManager(db, ner_model)

                        # 3. Execution
                        pb = st.progress(0)
                        st.toast("Vectorizing 28k+ entities... This will take a few minutes.", icon="🧠")

                        res = mm.rebuild_all_vectors(progress_callback=lambda x: pb.progress(x))

                        st.success(res)
                        st.balloons()

                        # 4. Refresh Page State
                        st.session_state.vec_ents = None # Clear cache
                        importlib.reload(src.database.db_manager) # Reload DB methods if needed
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed to rebuild memory: {e}")
                    st.code(str(e))
                    # Print stack trace for easier debugging
                    import traceback
                    st.code(traceback.format_exc())

        st.divider()

        kb_root = os.path.join(WORKSPACE_ROOT, "data", "knowledge_base")

        if os.path.exists(kb_root):
            # 1. Select Entity Type
            entity_types = [d for d in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, d))]
            entity_type = st.selectbox("Select Entity Type:", [""] + entity_types)

            if entity_type:
                entity_dir = os.path.join(kb_root, entity_type)

                # 2. Select File
                files = [f for f in os.listdir(entity_dir) if f.endswith('.txt') or f.endswith('.json')]
                selected_file = st.selectbox("Select File to Edit:", [""] + files)

                if selected_file:
                    file_path = os.path.join(entity_dir, selected_file)

                    # 3. Read & Edit
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            current_content = f.read()

                        st.markdown(f"**Editing:** `{os.path.join(entity_type, selected_file)}`")
                        new_content = st.text_area("File Content:", current_content, height=400)

                        if st.button("💾 Save Changes"):
                            # Validation for JSON
                            valid = True
                            if selected_file.endswith(".json"):
                                try:
                                    json.loads(new_content)
                                except json.JSONDecodeError as e:
                                    st.error(f"Invalid JSON Format: {e}")
                                    valid = False

                            if valid:
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(new_content)
                                st.success(f"Successfully saved {selected_file}!")
                                time.sleep(1)
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error reading file: {e}")
        else:
            st.error(f"Knowledge Base path not found at: {kb_root}")



