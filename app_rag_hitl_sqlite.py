import os
import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- LangChain imports (for your versions) ---
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ==================== CONFIG ====================
os.environ["OPENAI_API_KEY"] = "sk-proj-XXXX"  # replace with your key

DB_PATH = "rag_audit.db"
VECTOR_DIR = "rag_store"
DOC_PATH = "sample_docs.txt"

# ==================== DB FUNCTIONS ====================
def init_db():
    """Initialize the SQLite DB and create the table if needed."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rag_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        question TEXT,
        llm_answer TEXT,
        confidence REAL,
        final_answer TEXT,
        human_feedback TEXT,
        context_preview TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_log(question, answer, confidence, final_answer, human_feedback, explain):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO rag_audit_log (timestamp, question, llm_answer, confidence, final_answer, human_feedback, context_preview)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        question,
        answer,
        float(confidence),
        final_answer,
        human_feedback,
        explain["context_preview"]
    ))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM rag_audit_log ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# ==================== VECTORSTORE ====================
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DOC_PATH):
        with open(DOC_PATH, "w") as f:
            f.write(
                "LangChain is a framework for building LLM-based applications.\n"
                "It provides chains, memory, and retrieval tools.\n"
                "Chroma is a vector database for semantic search.\n"
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

prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the context below to answer the question clearly.
If the context does not provide enough information, say "I'm not confident about the answer."

Context:
{context}

Question: {question}

Answer:
""")

# ==================== RAG FUNCTION ====================
def get_rag_response(question: str):
    retrieved_docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join(
        [f"[Doc {i+1}] {getattr(doc, 'page_content', doc.get('page_content', ''))}" for i, doc in enumerate(retrieved_docs)]
    )

    response = llm.invoke(prompt.format(context=context_text, question=question))
    answer = response.content.strip()

    docs_with_scores = vectordb.similarity_search_with_score(question, k=3)
    scores = [score for _, score in docs_with_scores]
    avg_score = sum(scores) / len(scores) if scores else 0
    confidence = max(0, min(1, 1 - avg_score / 1.5))

    explainability = {"context_preview": context_text[:300] + "..." if len(context_text) > 300 else context_text}

    return answer, confidence, explainability

# ==================== STREAMLIT APP ====================
st.set_page_config(page_title="Explainable RAG with SQLite + HITL", layout="wide")
st.title("🧠 Explainable RAG with Confidence & HITL Logging (SQLite)")

tab1, tab2 = st.tabs(["💬 Ask a Question", "📊 Confidence Dashboard"])

init_db()

with tab1:
    question = st.text_area("Enter your question:", placeholder="e.g., What is LangChain?")
    if st.button("Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Generating RAG answer..."):
                answer, confidence, explain = get_rag_response(question)

            st.markdown("### 🧩 Explainability")
            st.info(explain["context_preview"])

            st.markdown("### 🤖 Model Answer")
            st.write(answer)

            st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

            if confidence < 0.5:
                st.warning("⚠️ Low confidence detected! Please review or edit the answer below.")
                human_feedback = st.text_area("Your feedback or correction:", value=answer)
                if st.button("✅ Approve/Submit Feedback"):
                    final_answer = human_feedback.strip() or answer
                    insert_log(question, answer, confidence, final_answer, human_feedback, explain)
                    st.success("✅ Feedback stored in SQLite database!")
            else:
                insert_log(question, answer, confidence, answer, "", explain)
                st.success("✅ Confident answer logged in database!")

with tab2:
    st.markdown("### Confidence Trend Visualization")
    df = get_logs()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["timestamp"], df["confidence"], marker="o", label="Confidence Score")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Confidence")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.markdown("### Recent Logs")
        st.dataframe(df.head(20))
    else:
        st.info("No logs found yet — try asking some questions!")
