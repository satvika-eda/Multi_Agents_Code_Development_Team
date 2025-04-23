from typing import TypedDict
import os
import re
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Command
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "Qwen/Qwen2.5-Coder-0.5B"

planner = base_model
developer = base_model
debugger = base_model
explainer = base_model
cot = base_model

developer_model = AutoModelForCausalLM.from_pretrained(
    developer,
    device_map="auto",
    torch_dtype=torch.float32,
)
developer_model.load_state_dict(torch.load("models/student_generator_model.pt"))
developer_tokenizer = AutoTokenizer.from_pretrained(developer)

cot_model = AutoModelForCausalLM.from_pretrained(
    cot,
    device_map="auto",
    torch_dtype=torch.float32,
)
cot_model.load_state_dict(torch.load("models/student_cot_model.pt"))
cot_tokenizer = AutoTokenizer.from_pretrained(cot)

planner_model = AutoModelForCausalLM.from_pretrained(
    planner,
    device_map="auto",
    torch_dtype=torch.float32,
)
planner_tokenizer = AutoTokenizer.from_pretrained(planner)

debugger_model = AutoModelForCausalLM.from_pretrained(
    debugger,
    device_map="auto",
    torch_dtype=torch.float32,
)
debugger_model.load_state_dict(torch.load("models/student_debugger_model.pt"))
debugger_tokenizer = AutoTokenizer.from_pretrained(debugger)

explainer_model = AutoModelForCausalLM.from_pretrained(
    explainer,
    device_map="auto",
    torch_dtype=torch.float32,
)
explainer_tokenizer = AutoTokenizer.from_pretrained(explainer)

print("Hello")

class AgentState(TypedDict):
    messages: list[BaseMessage]
    output: str
    is_final: bool

def my_code_tool(query: str) -> Command:
    print("code tool")
    prompt = (
        "You are a coding assistant. Generate only a markdown code block with valid Python code:\n"
        f"{query}"
    )
    inputs = developer_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = developer_model.generate(
            **inputs, max_new_tokens=512,
            pad_token_id=developer_tokenizer.eos_token_id,
        )
    full_text = developer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_text = full_text[len(prompt):]
    return Command(update={
        "output": full_text,
        "is_final": True,
        "messages": [ToolMessage(content=full_text, tool_call_id="my_code_tool_call")]
    })

def my_debug_tool(code: str) -> Command:
    prompt = f"You are a QA engineer. Analyze the Python code for errors:\n{code}"
    inputs = debugger_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = debugger_model.generate(
            **inputs, max_new_tokens=512,
            pad_token_id=debugger_tokenizer.eos_token_id,
        )
    full_text = debugger_tokenizer.decode(outputs[0], skip_special_tokens=True)
    suggestion = full_text[len(prompt):].strip()
    return Command(update={
        "output": suggestion,
        "is_final": True,
        "messages": [ToolMessage(content=suggestion, tool_call_id="my_debug_tool_call")]
    })

def my_explainer_tool(code: str) -> Command:
    prompt = f"You are a technical writer. Explain this Python code simply:\n{code}"
    inputs = explainer_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = explainer_model.generate(
            **inputs, max_new_tokens=512,
            pad_token_id=explainer_tokenizer.eos_token_id,
        )
    full_text = explainer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = full_text[len(prompt):].strip()
    return Command(update={
        "output": explanation,
        "is_final": True,
        "messages": [ToolMessage(content=explanation, tool_call_id="my_explainer_tool_call")]
    })

def developer_node(state: AgentState) -> AgentState:
    print("developer")
    query = state["output"]
    command = my_code_tool(query)
    new_state = {**state, **command.update}
    new_state["messages"] += command.update.get("messages", [])
    return new_state

def debugger_node(state: AgentState) -> AgentState:
    code = state.get("output", "")
    if not code:
        last_msg = state["messages"][-1].content
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", last_msg, re.DOTALL)
        code = match.group(1).strip() if match else last_msg.strip()
    command = my_debug_tool(code)
    new_state = {**state, **command.update}
    new_state["messages"] += command.update.get("messages", [])
    return new_state

def explainer_node(state: AgentState) -> AgentState:
    code = state.get("output", "")
    if not code:
        last_msg = state["messages"][-1].content
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", last_msg, re.DOTALL)
        code = match.group(1).strip() if match else last_msg.strip()
    command = my_explainer_tool(code)
    new_state = {**state, **command.update}
    new_state["messages"] += command.update.get("messages", [])
    return new_state

def cot_node(state: AgentState) -> AgentState:
    print("cot")
    user_input = state["messages"][-1].content
    prompt = (
        "You are a helpful assistant. Break down the problem step by step before coding.\n"
        f"Problem: {user_input}\n\nReasoning:"
    )
    inputs = cot_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = cot_model.generate(
            **inputs, max_new_tokens=512,
            pad_token_id=cot_tokenizer.eos_token_id,
        )
    reasoning = cot_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    state["messages"].append(ToolMessage(content=reasoning, tool_call_id="cot_node_call"))
    state["output"] = reasoning
    state["is_final"] = False
    return state

def classify_intent(prompt_text: str) -> str:
    classification_prompt = (
        "Classify this user message as one of:\n"
        "- developer (code writing)\n"
        "- debugger (find bugs)\n"
        "- explainer (explain code)\n"
        "- planner (general/non-code)\n\n"
        f"User message:\n{prompt_text}\n\n"
        "Return only one word:"
    )
    response = planner_model.generate(
        **planner_tokenizer(classification_prompt, return_tensors="pt"),
        max_new_tokens=10,
        pad_token_id=planner_tokenizer.eos_token_id,
    )
    output = planner_tokenizer.decode(response[0], skip_special_tokens=True)
    for intent in ["developer", "debugger", "explainer", "planner"]:
        if intent in output.lower():
            return intent
    return "planner"

def planner_router(state: AgentState):
    print("planner_router called")
    if state.get("is_final", False):
        return {"next_agent": "END"}
    user_msg = state["messages"][-1].content
    intent = classify_intent(user_msg)
    print("Intent:", intent)

    if intent == "planner":
        reply = (
            f"You're a coding assistant. The user said: '{user_msg}'\n\n"
            "Reply briefly: you only handle code writing, debugging, and explanation."
        )
        out = planner_model.generate(
            **planner_tokenizer(reply, return_tensors="pt"),
            max_new_tokens=100,
            pad_token_id=planner_tokenizer.eos_token_id,
        )
        msg = AIMessage(content=planner_tokenizer.decode(out[0], skip_special_tokens=True).strip())
        state["messages"].append(msg)
        state["output"] = msg.content
        state["is_final"] = True
        return {"next_agent": "END"}
    elif intent == "developer":
        return {"next_agent": "cot"}
    else:
        return {"next_agent": intent}

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_router)
    workflow.add_node("cot", cot_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("explainer", explainer_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        lambda state: state["next_agent"],
        {
            "cot": "cot",
            "debugger": "debugger",
            "explainer": "explainer",
            "END": END
        }
    )
    workflow.add_edge("cot", "developer")
    workflow.add_edge("developer", "planner")
    workflow.add_edge("debugger", "planner")
    workflow.add_edge("explainer", "planner")
    workflow.add_edge("planner", END)

    return workflow.compile()



initial_state = {
    "messages": [
        HumanMessage(content="""write code in python for first ten prime numbers""")
    ],
    "output": "",
    "is_final": False,
}

# app = create_workflow()

# result_state = app.invoke(initial_state)

# from IPython.display import Image, display
# from langgraph.graph.visualization import MermaidDrawMethod

# # Assume your compiled graph is stored in a variable called `app_graph`
# # For example:
# # app_graph = create_workflow()

# display(
#     Image(
#         app.draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API  # Uses mermaid.ink to render diagram
#         )
#     )
# )

# print("Final State:")
# print(result_state)
