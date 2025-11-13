import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command

# --- Streamlit setup ---
st.set_page_config(page_title="HITL Agent", page_icon="🤖", layout="wide")
st.title("🤖 Human-in-the-Loop Agent (LangChain 1.0.5)")

# --- API Key ---
os.environ["OPENAI_API_KEY"] = "sk-proj-*****"

# --- Define Tools ---
@tool
def write_file(filename: str, content: str) -> str:
    """Writes content to a file."""
    return f"[Simulated] Wrote '{filename}' with {len(content)} chars."

@tool
def execute_sql(query: str) -> str:
    """Executes a SQL query."""
    return f"[Simulated] Executed SQL: {query}"

@tool
def read_data(source: str) -> str:
    """Reads data from a source."""
    return f"[Simulated] Read data from {source}"

# --- Initialize agent (cached) ---
@st.cache_resource
def get_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(
        model=llm,
        tools=[write_file, execute_sql, read_data],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": True,
                    "execute_sql": {"allowed_decisions": ["approve", "reject", "edit"]},
                    "read_data": False,
                },
                description_prefix="Tool execution pending approval",
            ),
        ],
        checkpointer=InMemorySaver(),
    )

agent = get_agent()

# --- Session state setup ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-001"
if "interrupt" not in st.session_state:
    st.session_state.interrupt = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Step 1: Ask agent ---
if st.session_state.interrupt is None:
    with st.form("ask_agent"):
        st.subheader("💬 Ask the Agent")
        user_input = st.text_area("Enter your request:", "Execute SQL 'select * from users'")
        submitted = st.form_submit_button("Run Agent")

    if submitted and user_input.strip():
        with st.spinner("🤖 Agent thinking..."):
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        if "__interrupt__" in result:
            st.session_state.interrupt = result  # store interrupt
            st.rerun()
        else:
            st.success("✅ No human approval needed.")
            last_msg = result["messages"][-1]
            st.write(getattr(last_msg, "content", last_msg))

# --- Step 2: Handle interrupt & human decision ---
else:
    interrupt = st.session_state.interrupt["__interrupt__"][0]
    req = interrupt.value["action_requests"][0]
    tool_name = req.get("name")
    tool_args = req.get("args")

    st.warning("⚠️ Tool execution pending approval")
    st.write(f"**Tool:** `{tool_name}`")
    st.json(tool_args)

    with st.form("approval_form"):
        decision = st.radio("Your decision:", ["Approve", "Edit", "Reject"], index=0)
        edited_args = {}
        if decision == "Edit":
            st.write("🔧 Modify arguments:")
            for k, v in tool_args.items():
                edited_args[k] = st.text_input(f"{k}", value=str(v))

        confirm = st.form_submit_button("Submit Decision")

    if confirm:
        if decision == "Approve":
            final_decision = {"type": "approve"}
        elif decision == "Reject":
            final_decision = {"type": "reject"}
        else:
            final_decision = {
                "type": "edit",
                "edited_action": {"name": tool_name, "args": edited_args},
            }

        with st.spinner("🔄 Resuming agent..."):
            resume_cmd = Command(resume={"decisions": [final_decision]})
            resumed = agent.invoke(resume_cmd, config=config)

        # Clear interrupt from session
        st.session_state.interrupt = None

        st.success("✅ Final Output:")
        last_msg = resumed["messages"][-1]
        st.write(getattr(last_msg, "content", last_msg))
