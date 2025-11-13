import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command

# --- API Key ---
os.environ["OPENAI_API_KEY"] = "sk-proj-**********"


# --- Define tools ---
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


# --- Create agent ---
agent = create_agent(
    model="gpt-4o-mini",
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


# --- Run agent ---
thread_id = "thread-001"
config = {"configurable": {"thread_id": thread_id}}

print("🤖 Asking agent to perform task...\n")

#result = agent.invoke(
#    {"messages": [HumanMessage(content="Write 'Hello Human!' to test.txt")]},
#    config=config,
#)

result = agent.invoke(
    {"messages": [HumanMessage(content="Execute SQL 'select * from users' ")]},
    config=config,
)


# --- Handle Human-in-the-Loop interrupt ---
if "__interrupt__" in result:
    interrupt = result["__interrupt__"][0]
    req = interrupt.value["action_requests"][0]

    tool_name = req.get("name")
    tool_args = req.get("args")

    print(f"\n⚠️  Tool execution pending approval:")
    print(f"   🔧 Tool: {tool_name}")
    print(f"   🧩 Arguments: {tool_args}\n")

    # --- Ask user for input ---
    while True:
        decision_input = input("👉 Approve (A), Edit (E), or Reject (R)? ").strip().lower()
        if decision_input in ["a", "e", "r"]:
            break
        print("Please type A, E, or R.")

    if decision_input == "a":
        decision = {"type": "approve"}

    elif decision_input == "r":
        decision = {"type": "reject"}

    elif decision_input == "e":
        new_args = {}
        for k, v in tool_args.items():
            new_val = input(f"Enter new value for {k} [{v}]: ").strip() or v
            new_args[k] = new_val

        decision = {
            "type": "edit",
            "edited_action": {"name": tool_name, "args": new_args},
        }

    # --- Resume agent after decision ---
    print("\n🔄 Resuming agent with your decision...\n")
    resume_cmd = Command(resume={"decisions": [decision]})
    resumed = agent.invoke(resume_cmd, config=config)

    last_msg = resumed["messages"][-1]
    print("✅ Final Output:", getattr(last_msg, "content", last_msg))

else:
    print("✅ No human approval needed.")
    last_msg = result["messages"][-1]
    print(getattr(last_msg, "content", last_msg))
