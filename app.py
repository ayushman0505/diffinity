# app.py

import streamlit as st
from utils.text_extractor import extract_text_from_pdf, handle_uploaded_file, cleanup_temp_file
from utils.semantic_diff import compute_semantic_diff
from utils.summarizer import summarize_differences
from utils.qa_engine import SemanticQASystem
from utils.graph_drawer import draw_diff_graph

st.set_page_config(page_title="Diffinity", layout="wide")
st.title("🧠 Diffinity — Intelligent Document Comparator")
st.markdown("""
Compare two versions of a document to:
- 🔍 Detect semantic differences
- 📝 Generate a natural summary of changes
- 💬 Ask AI-powered questions about the document
- 📊 Visualize changes as a semantic graph
""")

# --- File Upload ---
st.sidebar.header("📄 Upload Documents")
file1 = st.sidebar.file_uploader("Upload Original PDF", type=["pdf"])
file2 = st.sidebar.file_uploader("Upload Revised PDF", type=["pdf"])

if file1 and file2:
    path1 = handle_uploaded_file(file1)
    path2 = handle_uploaded_file(file2)

    doc1_text = extract_text_from_pdf(path1)
    doc2_text = extract_text_from_pdf(path2)

    cleanup_temp_file(path1)
    cleanup_temp_file(path2)

    st.session_state['doc1'] = doc1_text
    st.session_state['doc2'] = doc2_text

    tabs = st.tabs(["📄 Diff Viewer", "🧠 Summary", "💬 Ask Questions", "📊 Visualize"])

    with tabs[0]:
        st.header("📄 Semantic Differences")
        diffs = compute_semantic_diff(doc1_text, doc2_text)
        st.session_state['diffs'] = diffs
        for label, text, score in diffs[:30]:
            st.markdown(f"**{label}** → _{text}_ (Score: {score:.2f})")

    with tabs[1]:
        st.header("🧠 AI Summary of Changes")
        if 'diffs' in st.session_state:
            summary = summarize_differences(st.session_state['diffs'])
            st.success(summary)

    with tabs[2]:
        st.header("💬 Ask Questions about the Document")
        qa_engine = SemanticQASystem()
        qa_engine.build_index(doc2_text)
        query = st.text_input("Ask something about the updated document:")
        if query:
            answer = qa_engine.ask(query)
            st.info(answer)

    with tabs[3]:
        st.header("📊 Graph View of Changes")
        if 'diffs' in st.session_state:
            draw_diff_graph(st.session_state['diffs'])
else:
    st.warning("Please upload both original and revised PDF files from the sidebar.")
