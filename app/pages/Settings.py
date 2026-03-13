import streamlit as st

st.set_page_config(page_title="Settings", layout="wide")

st.title("⚙️ System Settings")
st.markdown("Configure the **Memory & Intelligence** parameters for the Hybrid Predictor.")

# Initialize Session State Defaults if not present
if 'settings_vector_weight' not in st.session_state:
    st.session_state.settings_vector_weight = 0.5
if 'settings_fuzzy_weight' not in st.session_state:
    st.session_state.settings_fuzzy_weight = 0.5

with st.container(border=True):
    st.header("🧠 Intelligence Balance")
    st.caption("Determine how much the system should value 'Context' vs 'Identity'.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vector Context (The 'Where')")
        st.info("How much we trust the semantic context (e.g. typical phrases around a name).")
        v_weight = st.slider("Vector Weight", 0.0, 1.0, 
                             st.session_state.settings_vector_weight, 
                             key="v_slider")
        
    with col2:
        st.subheader("Fuzzy Identity (The 'Who')")
        st.info("How much we trust the spelling similarity to known names.")
        f_weight = st.slider("Fuzzy Weight", 0.0, 1.0, 
                             st.session_state.settings_fuzzy_weight, 
                             key="f_slider")

    # Update State
    st.session_state.settings_vector_weight = v_weight
    st.session_state.settings_fuzzy_weight = f_weight
    
    # Visualization of Balance
    total = v_weight + f_weight
    if total == 0: total = 1 # Avoid div/0
    
    st.divider()
    st.write("### Current Mix")
    
    c1, c2 = st.columns([v_weight, f_weight] if f_weight > 0 else [1, 0.01])
    c1.metric("Vector Influence", f"{v_weight/total:.1%}")
    if f_weight > 0:
        c2.metric("Fuzzy Influence", f"{f_weight/total:.1%}")

with st.container(border=True):
    st.header("🛠️ Advanced Thresholds")
    
    # Initialize Defaults
    if 'settings_vector_threshold' not in st.session_state:
        st.session_state.settings_vector_threshold = 0.70
    if 'settings_fuzzy_threshold' not in st.session_state:
        st.session_state.settings_fuzzy_threshold = 80

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Vector Similarity Threshold**")
        st.caption("Minimum semantic similarity (Cosine) to consider a context match.")
        v_thresh = st.slider("Vector Limit (0.0 - 1.0)", 0.5, 1.0, 
                             st.session_state.settings_vector_threshold, step=0.05,
                             key="v_thresh_slider")
        
    with col_b:
        st.write("**Fuzzy Match Threshold**")
        st.caption("Minimum spelling similarity (Ratio) to accept a known entity.")
        f_thresh = st.slider("Fuzzy Limit (0 - 100)", 50, 100, 
                             st.session_state.settings_fuzzy_threshold, step=5,
                             key="f_thresh_slider")

    # Update State
    st.session_state.settings_vector_threshold = v_thresh
    st.session_state.settings_fuzzy_threshold = f_thresh

st.success("Settings are applied automatically to the **Smart Review** session.")
