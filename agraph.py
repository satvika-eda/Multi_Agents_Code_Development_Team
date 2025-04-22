from typing import TypedDict
import os
import re
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Command
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

key = "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"

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = key

planner = ChatOpenAI(model="gpt-4o")

base_model = "Qwen/Qwen2.5-Coder-0.5B"

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

developer_tokenizer = AutoTokenizer.from_pretrained(
    developer,
)

cot_model = AutoModelForCausalLM.from_pretrained(
    debugger,
    device_map="auto",
    torch_dtype=torch.float32,
)

cot_model.load_state_dict(torch.load("models/student_cot_model.pt"))

cot_tokenizer = AutoTokenizer.from_pretrained(
    cot,
)

debugger_model = AutoModelForCausalLM.from_pretrained(
    debugger,
    device_map="auto",
    torch_dtype=torch.float32,
)

debugger_model.load_state_dict(torch.load("models/student_debugger_model.pt"))

debugger_tokenizer = AutoTokenizer.from_pretrained(
    debugger,
)

explainer_model = AutoModelForCausalLM.from_pretrained(
    explainer,
    device_map="auto",
    torch_dtype=torch.float32,
)

explainer_tokenizer = AutoTokenizer.from_pretrained(
    explainer,
)

class AgentState(TypedDict):
    messages: list[BaseMessage]
    output: str
    is_final: bool

def my_code_tool(query: str) -> Command:
    """
    Given a query string, generate Python code.
    Returns a Command update with:
      - "output": The generated code.
      - "is_final": True (final output flag).
      - "messages": A list with a ToolMessage containing the code.
    """
    print("code tool")
    prompt = (
        "You are a coding assistant. When given a prompt, generate only a markdown code block that contains "
        "valid Python code. The code block must start with ```python on its own line, then include the code, and "
        "finally end with ``` on its own line. Do not include any extra text, comments, or explanations.\n"
        f"{query}"
    )
    inputs = developer_tokenizer(prompt, return_tensors="pt")
    inputs.to("mps")

    with torch.no_grad():
        outputs = developer_model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=developer_tokenizer.eos_token_id,
        )

    full_text = developer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)
    # full_text = response.content
    full_text = full_text[len(prompt):]

    # code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", full_text, re.DOTALL)
    # if code_block_match:
    #     code = code_block_match.group(1).strip()
    # else:
    #     code = full_text.strip()
    print("my_code_tool was called")
    print(full_text)
    return Command(
        update={
            "output": full_text,
            "is_final": True,
            "messages": [ToolMessage(content=full_text, tool_call_id="my_code_tool_call")],
        }
    )

def my_debug_tool(code: str) -> Command:
    """
    Analyze the given Python code for errors and suggest fixes.
    Returns a Command update with the suggestions, sets is_final True.
    """
    prompt = (
        "You are a QA engineer. Analyze the following Python code for errors and suggest fixes:\n"
        f"{code}"
    )
    inputs = debugger_tokenizer(prompt, return_tensors="pt")
    inputs.to("mps")

    with torch.no_grad():
        outputs = debugger_model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=debugger_tokenizer.eos_token_id,
        )

    full_text = debugger_tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_text = full_text[len(prompt):]
    suggestion = full_text.strip()
    print("my_debug_tool was called")
    return Command(
        update={
            "output": suggestion,
            "is_final": True,
            "messages": [ToolMessage(content=suggestion, tool_call_id="my_debug_tool_call")],
        }
    )

def my_explainer_tool(code: str) -> Command:
    """
    Explain the given Python code in clear and simple language.
    Returns a Command update with the explanation, sets is_final True.
    """
    prompt = (
        "You are a technical writer. Explain the following Python code in clear, simple language:\n"
        f"{code}"
    )
    inputs = explainer_tokenizer(prompt, return_tensors="pt")
    inputs.to("mps")

    with torch.no_grad():
        outputs = explainer_model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=explainer_tokenizer.eos_token_id,
        )

    full_text = explainer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_text = full_text[len(prompt):]
    explanation = full_text.strip()
    print("my_explainer_tool was called")
    return Command(
        update={
            "output": explanation,
            "is_final": True,
            "messages": [ToolMessage(content=explanation, tool_call_id="my_explainer_tool_call")],
        }
    )


def developer_node(state: AgentState) -> AgentState:
    """
    Extract the user query from the last message and call my_code_tool.
    Merge the tool's Command update into the state.
    """
    print("developer")
    query = state["messages"][-1].content
    command = my_code_tool(query)
    new_state = {**state, **command.update}
    new_state["messages"] = state["messages"] + command.update.get("messages", [])
    return new_state

def debugger_node(state: AgentState) -> AgentState:
    """
    Use the current state's output (assumed to be code) as input for debugging.
    Calls my_debug_tool to analyze the code and merge its results.
    """
    code = state.get("output", "")
    # If no code is present in state, try to extract from the last human message.
    if not code:
        last_msg = state["messages"][-1].content
        code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", last_msg, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
        else:
            code = last_msg.strip()
    command = my_debug_tool(code)
    new_state = {**state, **command.update}
    new_state["messages"] = state["messages"] + command.update.get("messages", [])
    return new_state

def explainer_node(state: AgentState) -> AgentState:
    """
    Use the current state's output (if available) or extract a code block from the last message.
    Calls my_explainer_tool to produce an explanation and merge its result.
    """
    code = state.get("output", "")
    if not code:
        # Try to get a code block from the last human message.
        last_msg = state["messages"][-1].content
        code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", last_msg, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1).strip()
        else:
            code = last_msg.strip()
    command = my_explainer_tool(code)
    new_state = {**state, **command.update}
    new_state["messages"] = state["messages"] + command.update.get("messages", [])
    return new_state

def classify_intent(prompt_text: str) -> str:
    """
    Use the model to determine the intent of the prompt.
    The assistant should return one of the following words exactly:
    "debugger", "explainer", "developer", or "planner".
    
    If the user message is general (not about code), respond with 'planner'.
    """
    classification_prompt = (
        "You are a helpful assistant that routes user coding questions to specialized agents.\n"
        "Your job is to classify the user's prompt into one of the following categories:\n"
        "- 'debugger' → if the prompt is asking to find and fix bugs in code.\n"
        "- 'explainer' → if the prompt is asking for an explanation of code.\n"
        "- 'developer' → if the prompt is asking to write new code or make changes.\n"
        "- 'planner' → if the prompt is general (e.g. greetings, casual questions, or anything not related to code).\n\n"
        "User prompt:\n"
        f"{prompt_text}\n\n"
        "Return only one word: debugger, explainer, developer, or planner."
    )
    
    classification_response = planner.invoke([HumanMessage(content=classification_prompt)])
    intent = classification_response.content.strip().lower()
    
    # Validate output
    if intent in ["debugger", "explainer", "developer", "planner"]:
        return intent
    return "planner"  # safe fallback

def planner_router(state: AgentState):
    print("planner_router called, is_final:", state.get("is_final"))
    if state.get("is_final", False):
        return {"next_agent": "END"}
    
    # Use the model to classify the intent based on the last message.
    last_msg = state["messages"][-1].content
    intent = classify_intent(last_msg)
    print("Classified intent as:", intent)

    if intent == "planner":
        last_user_message = state["messages"][-1].content
        redirect_prompt = (
            f"The user said: '{last_user_message}'\n\n"
            "You're a coding assistant. If the message is not related to programming (like writing code, debugging, or explanation), "
            "then politely reply that you only support coding-related queries. Keep it short and to the point. "
            "Do not try to answer general questions. You can chit-chat and greet. Also redirect clearly if different topics are asked."
        )
        response = planner.invoke([HumanMessage(content=redirect_prompt)])
        msg = AIMessage(content=response.content.strip())
        state["messages"].append(msg)
        state["output"] = msg.content
        state["is_final"] = True
        return {"next_agent": "END"}
    
    return {"next_agent": intent}

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_router)
    workflow.add_node("developer", developer_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("explainer", explainer_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        lambda state: state["next_agent"],
        {
            "developer": "developer",
            "debugger": "debugger",
            "explainer": "explainer",
            "END": END
        }
    )
    workflow.add_edge("developer", "planner")
    workflow.add_edge("debugger", "planner")
    workflow.add_edge("explainer", "planner")
    workflow.add_edge("planner", END)
    return workflow.compile()

# initial_state = {
#     "messages": [
#         HumanMessage(content="""write code in python for first ten prime numbers""")
#     ],
#     "output": "",
#     "is_final": False,
# }

# result_state = create_workflow().invoke(initial_state)
# print("Final State:")
# print(result_state)
