# langchain_structured_math.py
import os
from typing import List

# Set API key (you already set this in your env)
os.environ["OPENAI_API_KEY"] = "sk-proj-**********"

# LangChain imports (v1.0.5)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel  # ensure pydantic v1 compatibility

# Define the same schema using langchain_core.pydantic_v1.BaseModel
class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: List[Step]
    final_answer: str

# Create a ChatOpenAI instance pointed at the same model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Wrap the LLM to produce structured output matching MathReasoning
# strict=True enforces schema restrictions; set to False if you want more leniency
structured_llm = llm.with_structured_output(MathReasoning, strict=False)


# Compose the prompt (combine system + user roles into a single prompt string)
prompt = (
    "You are a helpful math tutor. Guide the user through the solution step by step.\n\n"
    "User: how can I solve 8x + 7 = -23"
)

# Invoke the structured LLM
# This returns an instance of MathReasoning (a pydantic model)
result: MathReasoning = structured_llm.invoke(prompt)

# Print the parsed & validated structured response
print("Parsed structured output (pydantic model):")
print(result)            # prints the MathReasoning object
print("\nFinal answer:", result.final_answer)
print("\nSteps:")
for i, s in enumerate(result.steps, start=1):
    print(f"Step {i}: explanation={s.explanation!r}, output={s.output!r}")
