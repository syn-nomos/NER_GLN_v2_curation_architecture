import streamlit as st

import pandas as pd

import plotly.express as px

import sys

import os

import json

import sqlite3



# Path resolution

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# app/pages -> app -> to_move

TO_MOVE_ROOT = os.path.dirname(os.path.dirname(APP_DIR))

ROOT_DIR_REAL = os.path.dirname(TO_MOVE_ROOT) # Actual workspace root if needed



if TO_MOVE_ROOT not in sys.path:

    sys.path.insert(0, TO_MOVE_ROOT)



from src.database.db_manager import DBManager



st.set_page_config(page_title="Dataset Analytics", page_icon="📊", layout="wide")



# --- HELPER: GET ACTIVE DB PATH ---

def resolve_db_path():

    # 1. Check Session State (Best source of truth if app is running)

    if "db" in st.session_state and hasattr(st.session_state.db, "db_path"):

        return st.session_state.db.db_path

        

    # 2. Check Session Config File (Persisted from Data_Loader)

    config_file = os.path.join(TO_MOVE_ROOT, "session_config.json")

    if os.path.exists(config_file):

        try:

            with open(config_file, 'r') as f:

                data = json.load(f)

                last = data.get("last_db_path")

                if last and os.path.exists(last):

                    return last

        except:

            pass

            

    # 3. Fallback to default locations

    default_1 = os.path.join(TO_MOVE_ROOT, "data", "annotations.db")

    if os.path.exists(default_1): return default_1

    

    default_2 = os.path.join(TO_MOVE_ROOT, "data", "db", "annotations.db")

    if os.path.exists(default_2): return default_2

    

    return None



# --- CUSTOM CSS ---

st.markdown("""

<style>

    .metric-card {

        background-color: #f0f2f6;

        border-radius: 10px;

        padding: 20px;

        text-align: center;

        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);

    }

    .metric-value {

        font_size: 2.5em;

        font-weight: bold;

        color: #31333F;

    }

    .metric-label {

        font_size: 1.1em;

        color: #666;

    }

</style>

""", unsafe_allow_html=True)



db_path = resolve_db_path()



st.title("📊 Knowledge Base Analytics")

if db_path:

    st.caption(f"📂 Active Database: `{os.path.basename(db_path)}`")

else:

    st.error("No Database Found. Please load one in 'Data Loader'.")

    st.stop()





# --- LOAD DATA ---

@st.cache_data(ttl=60)

def load_data(path_to_db):

    conn = sqlite3.connect(path_to_db)

    

    # 1. Inspect Columns to avoid "no such column" error

    # We want these columns

    desired_cols = [

        "id", "label", "text_span", 

        "confidence", "source_agent",

        "is_accepted", "is_rejected", "created_at"

    ]

    

    # Helper to clean query

    cur = conn.cursor()

    cur.execute("PRAGMA table_info(annotations)")

    existing_cols = {row[1] for row in cur.fetchall()}

    

    # Build safe SELECT

    select_parts = ["id", "label", "text_span", "LENGTH(text_span) as length"]

    

    for col in ["confidence", "source_agent", "is_accepted", "is_rejected", "created_at"]:

        if col in existing_cols:

            select_parts.append(col)

        else:

            select_parts.append(f"NULL as {col}") # Return NULL if missing

            

    query = f"SELECT {', '.join(select_parts)} FROM annotations"

    

    df_anns = pd.read_sql_query(query, conn)

    

    # 2. Sentences Stats

    # Check for is_flagged

    cur.execute("PRAGMA table_info(sentences)")

    sent_cols = {row[1] for row in cur.fetchall()}

    

    sent_select = ["id", "LENGTH(text) as length", "status", "dataset_split"]

    if "is_flagged" in sent_cols:

        sent_select.append("is_flagged")

    else:

        sent_select.append("0 as is_flagged")

        

    df_sents = pd.read_sql_query(f"SELECT {', '.join(sent_select)} FROM sentences", conn)

    

    conn.close()

    

    # --- DATA NORMALIZATION ---

    if not df_anns.empty:

        # Create a normalized label for statistics (handles Case + Separator mismatch)

        # Strategy: UPPERCASE and convert '_' to '-' (Standardizing to LEG-REFS style)

        df_anns['original_label'] = df_anns['label']

        df_anns['label_norm'] = df_anns['label'].fillna('UNK').astype(str).str.upper().str.replace('_', '-')

    

    return df_anns, df_sents



try:

    with st.spinner("Crunching numbers..."):

        df_anns, df_sents = load_data(db_path)

except Exception as e:

    st.error(f"Error loading database: {e}")

    st.stop()



# --- KPI ROW ---

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric("Total Sentences", f"{len(df_sents):,}")

with col2:

    st.metric("Total Annotations", f"{len(df_anns):,}")

with col3:

    if not df_anns.empty: 

        unique_ents = df_anns['text_span'].nunique()

    else: unique_ents = 0

    st.metric("Unique Entities", f"{unique_ents:,}")

with col4:

    reviewed = len(df_sents[df_sents['status'] == 'reviewed'])

    pct = (reviewed / len(df_sents) * 100) if len(df_sents) > 0 else 0

    st.metric("Reviewed Progress", f"{pct:.1f}%", f"{reviewed} sent.")

with col5:

    flagged = df_sents['is_flagged'].sum()

    st.metric("Flagged Issues", f"{flagged}", delta_color="inverse")



st.divider()



# --- TABS ---

tab_dist, tab_quality, tab_corpus, tab_detailed, tab_fragments = st.tabs([

    "📈 Label Distribution", 

    "✅ Quality & Agents", 

    "📚 Corpus Stats",

    "🔎 Entity Explorer",

    "🧹 Fragments Inspection"

])



# --- TAB 1: DISTRIBUTION ---

with tab_dist:

    c1, c2 = st.columns([2, 1])

    

    with c1:

        st.subheader("Annotations by Entity Type")

        

        # Check for Label Inconsistencies

        if not df_anns.empty:

            inconsistent = df_anns.groupby('label_norm')['original_label'].nunique()

            inconsistent = inconsistent[inconsistent > 1]

            

            if not inconsistent.empty:

                st.warning(f"⚠️ Found Inconsistent Label Naming! (Merged in statistics)")

                with st.expander("See Inconsistencies"):

                    for norm, count in inconsistent.items():

                        variations = df_anns[df_anns['label_norm'] == norm]['original_label'].unique()

                        st.write(f"**{norm}** appears as: `{list(variations)}`")

        

        if not df_anns.empty:

            # Use NORMALIZED label for charts

            label_counts = df_anns['label_norm'].value_counts().reset_index()

            label_counts.columns = ['Label', 'Count']

            fig_bar = px.bar(

                label_counts, x='Label', y='Count', 

                color='Label', 

                text='Count',

                title="Entity Distribution (Normalized)",

            )

            fig_bar.update_layout(showlegend=False)

            st.plotly_chart(fig_bar, use_container_width=True)

            

            # Pie Chart

            fig_pie = px.pie(label_counts, values='Count', names='Label', hole=0.4, title="Relative Proportions")

            st.plotly_chart(fig_pie, use_container_width=True)

        else:

            st.info("No annotations found.")



    with c2:

        st.subheader("Top 20 Frequent Entities")

        if not df_anns.empty:

            top_ents = df_anns['text_span'].value_counts().head(20).reset_index()

            top_ents.columns = ['Entity Text', 'Frequency']

            st.dataframe(top_ents, use_container_width=True, height=600)



# --- TAB 2: QUALITY & AGENTS ---

with tab_quality:

    st.subheader("Annotation Status & Agent Performance")

    

    q1, q2 = st.columns(2)

    

    with q1:

        st.markdown("#### ✅ Acceptance Status")

        # Logic: Accepted=True, Rejected=True, else Pending

        if not df_anns.empty:

            def get_status(row):

                if row.get('is_accepted'): return 'Accepted'

                if row.get('is_rejected'): return 'Rejected'

                return 'Pending'

            

            df_anns['status_calc'] = df_anns.apply(get_status, axis=1)

            status_counts = df_anns['status_calc'].value_counts().reset_index()

            status_counts.columns = ['Status', 'Count']

            

            fig_stat = px.pie(status_counts, values='Count', names='Status', 

                             color='Status',

                             color_discrete_map={'Accepted':'#00CC96', 'Rejected':'#EF553B', 'Pending':'#FECB52'})

            st.plotly_chart(fig_stat, use_container_width=True)

        else:

            st.warning("No data.")

    

    with q2:

        st.markdown("#### 🤖 Source Agent Contribution")

        if not df_anns.empty:

            # Replace None with "Manual/Unknown"

            df_anns['source_agent'] = df_anns['source_agent'].fillna('Manual/Unknown')

            

            agent_counts = df_anns['source_agent'].value_counts().reset_index()

            agent_counts.columns = ['Agent', 'Count']

            

            fig_ag = px.bar(agent_counts, x='Count', y='Agent', orientation='h', title="Annotations per Source")

            st.plotly_chart(fig_ag, use_container_width=True)



    st.divider()

    st.markdown("#### 📏 Confidence Analysis (Model Predictions)")

    # Check if confidence column actually has data (it might be NULL from our Safe Select)

    if not df_anns.empty and 'confidence' in df_anns.columns and df_anns['confidence'].notna().any():

        fig_hist = px.histogram(df_anns, x='confidence', nbins=50, color='label', title="Confidence Score Distribution")

        st.plotly_chart(fig_hist, use_container_width=True)

    else:

        st.info("No confidence scores available in this Database version.")



# --- TAB 3: CORPUS STATS ---

with tab_corpus:

    st.subheader("Sentence Lengths & Density")

    

    k1, k2 = st.columns(2)

    

    with k1:

        st.caption("Distribution of Sentence Lengths (Characters)")

        fig_ln = px.histogram(df_sents, x='length', nbins=100, title="Sentence Character Count")

        st.plotly_chart(fig_ln, use_container_width=True)

        

    with k2:

        st.caption("Sentences by Split (Train/Test/Val)")

        split_counts = df_sents['dataset_split'].fillna('Unassigned').value_counts().reset_index()

        split_counts.columns = ['Split', 'Count']

        fig_spl = px.bar(split_counts, x='Split', y='Count', color='Split')

        st.plotly_chart(fig_spl, use_container_width=True)

        

# --- TAB 4: DETAILED EXPLORER ---

with tab_detailed:

    st.subheader("🔎 Deep Dive")

    st.markdown("Filter rows to inspect specific data subsets.")

    

    if not df_anns.empty:

        # Filters

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:

            sel_label = st.multiselect("Filter by Label (Normalized)", options=df_anns['label_norm'].unique())

        with f_col2:

            sel_agent = st.multiselect("Filter by Source Agent", options=df_anns['source_agent'].unique())

        with f_col3:

            show_raw = st.checkbox("Show Raw Labels")

            

        # Apply Filter

        df_show = df_anns.copy()

        if sel_label:

            df_show = df_show[df_show['label_norm'].isin(sel_label)]

        if sel_agent:

            df_show = df_show[df_show['source_agent'].isin(sel_agent)]

        

        # Display Cols

        cols_to_show = ['id', 'label_norm', 'text_span', 'confidence', 'source_agent']

        if show_raw:

             cols_to_show.insert(1, 'original_label')

             

        st.write(f"Showing {len(df_show)} annotations matching filters.")

        st.dataframe(df_show[cols_to_show], use_container_width=True)

    else:

        st.info("No data to explore.")



# --- TAB 5: FRAGMENTS ---

with tab_fragments:

    st.subheader("π§Ή Fragment Cleaner")

    st.markdown("Identifies potential 'garbage' annotations like single characters, punctuation, or very short tokens.")

    

    if not df_anns.empty:

        col_frag1, col_frag2 = st.columns([1, 2])

        

        with col_frag1:

            frag_mode = st.radio("Detection Mode", ["Length < X", "Punctuation/Symbols", "Single Token Digits"])

            

            fragment_df = pd.DataFrame()

            

            if frag_mode == "Length < X":

                thresh = st.slider("Max Character Length", 1, 6, 2)

                # Ensure string type

                fragment_df = df_anns[df_anns['text_span'].fillna('').astype(str).str.len() <= thresh]

                

            elif frag_mode == "Punctuation/Symbols":

                # Regex for non-alphanumeric (matches only if the whole string is non-alphanumeric)

                # \W matches non-word characters. We want strings that consist ENTIRELY of \W.

                fragment_df = df_anns[df_anns['text_span'].fillna('').astype(str).str.match(r'^[\W_]+$')]

                

            elif frag_mode == "Single Token Digits":

                # Matches "1", "20", "5" etc. often mistakes for dates/refs if context is lost

                fragment_df = df_anns[df_anns['text_span'].fillna('').astype(str).str.match(r'^\d+$')]



        with col_frag2:

            st.metric("Suspect Fragments Found", len(fragment_df))

            if not fragment_df.empty:

                st.dataframe(fragment_df[['id', 'text_span', 'label', 'confidence', 'source_agent']], use_container_width=True, height=400)

                

                # Bulk Action

                st.warning("β οΈ Dangerous Zone")

                if st.button(f"π—‘οΈ DELETE ALL {len(fragment_df)} SHOWN FRAGMENTS"):

                    ids_to_del = fragment_df['id'].tolist()

                    if ids_to_del:

                        try:

                             # Direct DB connection for deletion

                             conn_del = sqlite3.connect(db_path)

                             cur_del = conn_del.cursor()

                             placeholders = ','.join('?' for _ in ids_to_del)

                             cur_del.execute(f"DELETE FROM annotations WHERE id IN ({placeholders})", ids_to_del)

                             conn_del.commit()

                             conn_del.close()

                             st.success(f"Deleted {len(ids_to_del)} fragments! Reloading...")

                             st.rerun()

                        except Exception as e:

                            st.error(f"Deletion failed: {e}")

            else:

                st.success("No fragments found with current filters.")

