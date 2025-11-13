import os
import csv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# =============== 1️⃣ SETUP ==================
os.environ["OPENAI_API_KEY"] = "sk-proj-XXXX"  # replace with your key

LOG_FILE = "rag_audit_log.csv"
VECTOR_DIR = "rag_store"
DOC_PATH = "sample_docs.txt"

# =============== 2️⃣ INITIALIZATION ===============
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DOC_PATH):
        with open(DOC_PATH, "w") as f:
            f.write(
                "LangChain is a framework to build LLM-powered applications.\n"
                "It provides tools for memory, prompt templates, agents, and RAG.\n"
                "Chroma is a vector database used for retrieval and semantic search.\n"
                "RAG stands for Retrieval-Augmented Generation.\n"
            )

    loader = TextLoader(DOC_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=VECTOR_DIR)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return vectordb, retriever

vectordb, retriever = load_vectorstore()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the context below to answer the question clearly.
If context is insufficient, say "I'm not confident about the answer."

Context:
{context}

Question: {question}

Answer:
""")

# =============== 3️⃣ RAG FUNCTION ===============
def get_rag_response(question: str):
    retrieved_docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join(
        [f"[Doc {i+1}] {getattr(doc, 'page_content', doc.get('page_content', ''))}" for i, doc in enumerate(retrieved_docs)]
    )
    response = llm.invoke(rag_prompt.format(context=context_text, question=question))
    answer = response.content.strip()

    docs_with_scores = vectordb.similarity_search_with_score(question, k=3)
    scores = [score for _, score in docs_with_scores]
    avg_score = sum(scores) / len(scores) if scores else 0
    confidence = max(0, min(1, 1 - avg_score / 1.5))

    explainability = {
        "context_preview": context_text[:300] + "..." if len(context_text) > 300 else context_text
    }

    return answer, confidence, explainability

# =============== 4️⃣ LOGGING FUNCTION ===============
def log_interaction(question, answer, confidence, final_answer, human_feedback, explain):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "question", "llm_answer", "confidence",
                "final_answer", "human_feedback", "context_preview"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            question,
            answer,
            round(confidence, 3),
            final_answer,
            human_feedback,
            explain["context_preview"]
        ])

# =============== 5️⃣ STREAMLIT UI ===============
st.set_page_config(page_title="Explainable RAG with HITL", layout="wide")
st.title("🧠 Explainable RAG with Confidence & Human-in-the-Loop")

tab1, tab2 = st.tabs(["💬 Ask a Question", "📊 Confidence Trends"])

with tab1:
    question = st.text_area("Enter your question:", placeholder="e.g., What is LangChain?")
    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Retrieving and reasoning..."):
                answer, confidence, explain = get_rag_response(question)

            st.markdown("### 🧩 Explainability")
            st.info(explain["context_preview"])

            st.markdown("### 🤖 Model Answer")
            st.write(answer)

            st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

            if confidence < 0.5:
                st.warning("⚠️ Low confidence detected! Please review the answer below.")
                human_feedback = st.text_area("Human Feedback (edit or approve):", value=answer)
                if st.button("✅ Approve/Submit Feedback"):
                    final_answer = human_feedback.strip() or answer
                    log_interaction(question, answer, confidence, final_answer, human_feedback, explain)
                    st.success("✅ Feedback logged successfully!")
            else:
                log_interaction(question, answer, confidence, answer, "", explain)
                st.success("✅ Confident answer logged.")

with tab2:
    st.markdown("### Confidence Trend Visualization")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df["timestamp"], df["confidence"], marker="o", label="Model Confidence")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(0, 1)
            ax.set_title("Confidence Trends Over Time")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.dataframe(df.tail(10).sort_values("timestamp", ascending=False))
        else:
            st.info("No log data yet. Ask a few questions first.")
    else:
        st.info("No log file found. Ask a few questions to generate logs.")
