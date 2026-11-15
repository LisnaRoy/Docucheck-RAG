import tempfile
from pathlib import Path

import streamlit as st

from docucheck.pipeline import ask
from docucheck.retriever import index_pdf

st.set_page_config(page_title="DocuCheck", page_icon="📄")
st.title("DocuCheck")
st.caption("Ask questions about your PDF and get answers with page citations.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None and st.session_state.get("indexed_file") != uploaded.name:
    with st.spinner(f"Indexing {uploaded.name}..."):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pdf_path = Path(tmp) / uploaded.name
            pdf_path.write_bytes(uploaded.getvalue())
            try:
                index_pdf(pdf_path)
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state["indexed_file"] = uploaded.name
                st.session_state["messages"] = []

if "indexed_file" in st.session_state:
    st.caption(f"Indexed: {st.session_state['indexed_file']}")
else:
    st.caption("No upload yet. Questions use the index already saved in data/index.")


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['doc_name']}, page {s['page']}**")
            st.write(s["text"])


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a question about the document")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = ask(question)
        except FileNotFoundError:
            st.warning("No index yet. Upload a PDF first.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()
        st.write(result["answer"])
        if result["found"]:
            show_sources(result["sources"])

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"] if result["found"] else [],
        }
    )