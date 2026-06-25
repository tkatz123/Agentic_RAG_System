import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import build_graph

st.markdown("""
      <style>
      .stApp {
          background-color: #d0ebff;
          color: #000000;
      }
      .stTextInput input {
          background-color: #FFFFFF;
      }
      </style>
  """, unsafe_allow_html=True)



st.title("Agentic RAG Research Assistant 🤖")
st.write("Ask questions about AI engineering roles and AI/ML research papers.")

question = st.text_input("Ask a question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        agent = build_graph()
        result = agent.invoke({
            "question": question,
            "query": question,
            "docs": [],
            "relevant": False,
            "answer": "",
            "retries": 0
        })

        sources = []

        for doc in result['docs']:
            if doc.metadata.get("source") == "job_posting":
                sources.append(doc.metadata['company'])
            else:
                sources.append(doc.metadata['title'])
        
        sources = list(set(sources))

        st.subheader("Answer")
        st.write(result['answer'])
        
        st.subheader("Sources")
        for source in sources:
            st.write(f"- {source}")
