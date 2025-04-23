import streamlit as st
from graph import create_workflow
from langchain_core.messages import HumanMessage, AIMessage
import copy
import json
import os
import time


def log_ab_feedback(user_message, response_A, response_B, selected):
    """
    Log the A/B testing results including the user message,
    both responses, and which one was selected.
    """
    log_file = "ab_feedback_log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = {}

    timestamp = str(int(time.time()))
    logs[timestamp] = {
        "user_message": user_message,
        "qwen7B": response_A,
        "qwen0.5B": response_B,
        "selected": selected
    }
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

def generate_ab_responses(workflow, messages):
    """
    Generate two candidate responses using the workflow.
    We assume that invoking the workflow twice will produce
    slightly different responses due to the inherent stochasticity.
    """
    config = {"recursion_limit": 100}
    
    # First response
    response_1 = workflow.invoke({"messages": messages}, config)
    # Deep copy messages to avoid unexpected mutation.
    messages_copy = copy.deepcopy(messages)
    # Second response, potentially different than the first.
    response_2 = workflow.invoke({"messages": messages_copy}, config) 
    
    # Assuming response["messages"] is a list and the last message is the AI response.
    candidate_response_1 = response_1["messages"][-1]
    candidate_response_2 = response_2["messages"][-1]
    
    return candidate_response_1, candidate_response_2

def display_ab_test(candidate_A, candidate_B):
    """
    Display two candidate responses side by side and let user choose their preferred one.
    """
    st.markdown("### Which response do you prefer?")
    col_A, col_B = st.columns(2)

    selected = None  # Will capture the user's selection

    with col_A:
        st.markdown("**Response A:**")
        st.markdown(candidate_A.content)
        if st.button("Select qwen7B", key="select_7B"):
            selected = "qwen7B"

    with col_B:
        st.markdown("**Response B:**")
        st.markdown(candidate_B.content)
        if st.button("Select qwen0.5B", key="select_0.5B"):
            selected = "qwen0.5B"
    
    return selected

def process_user_input_ab(prompt):
    """Handles user input and A/B tests two responses from the workflow."""
    # Add user message to conversation
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Generate two candidate responses using the workflow
    candidate_A, candidate_B = generate_ab_responses(st.session_state.workflow, st.session_state.messages)
    
    # Store candidate responses in session state for use after selection
    st.session_state.candidate_A = candidate_A
    st.session_state.candidate_B = candidate_B
    st.session_state.last_user_message = prompt

def update_conversation_with_selected(selected):
    """Update the conversation based on the user's selected response."""
    if selected == "A":
        chosen = st.session_state.candidate_A
    else:
        chosen = st.session_state.candidate_B
    
    # Log A/B feedback
    log_ab_feedback(
        st.session_state.last_user_message,
        st.session_state.candidate_A.content,
        st.session_state.candidate_B.content,
        selected
    )
    
    # Append the chosen response to the conversation history
    st.session_state.messages.append(chosen)
    
    # Clean up the temporary candidate responses
    st.session_state.pop("candidate_A", None)
    st.session_state.pop("candidate_B", None)
    st.session_state.pop("last_user_message", None)

def display_chat_history():
    """Display the conversation history using st.chat_message."""
    for msg in st.session_state.messages:
        # Don't render temporary A/B test candidates
        if "candidate_A" in st.session_state and msg == st.session_state.candidate_A:
            continue
        if "candidate_B" in st.session_state and msg == st.session_state.candidate_B:
            continue
        
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

# ------------------------------
# Main App Layout and Logic
# ------------------------------

st.title("LangGraph Chatbot with A/B Testing")

# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow" not in st.session_state:
    st.session_state.workflow = create_workflow()

# Display chat history first
display_chat_history()

# Check if we are currently in A/B testing mode
if "candidate_A" in st.session_state and "candidate_B" in st.session_state:
    # Display two candidate responses and let the user pick one
    selected = display_ab_test(st.session_state.candidate_A, st.session_state.candidate_B)
    if selected:
        update_conversation_with_selected(selected)
        st.rerun()
else:
    # Standard conversation input area when not in A/B testing mode
    user_prompt = st.chat_input("Type your message")
    if user_prompt:
        process_user_input_ab(user_prompt)
        st.rerun()
