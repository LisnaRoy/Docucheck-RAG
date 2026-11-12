import streamlit as st

from docucheck.pipeline import ask

st.set_page_config(page_title="DocuCheck", page_icon="📄")
st.title("DocuCheck")
st.caption("Ask questions about your PDF and get answers with page citations.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None:
    st.session_state["pdf_name"] = uploaded.name
    st.info(
        f"Loaded {uploaded.name}. Indexing is not connected yet; "
        "answers come from placeholder text until the retriever is plugged in."
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []


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
        with st.spinner("Thinking..."):
            result = ask(question)
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