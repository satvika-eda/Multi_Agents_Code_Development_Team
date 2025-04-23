import streamlit as st
from graph import create_workflow
import json, os, time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Command

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

def log_feedback_data(feedback, messages):
    feedback_file = "feedback_log.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            logs = json.load(f)
    else:
        logs = {}
    
    timestamp = str(int(time.time()))
    logs[timestamp] = {
        "feedback": feedback,
        "messages": [ {"role": msg.type, "content": msg.content} for msg in messages ]
    }
    
    with open(feedback_file, "w") as f:
        json.dump(logs, f, indent=2)

def display_chat_history():
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg.type):
            st.markdown(msg.content)
            
        # Display feedback buttons only for the latest bot response
        if isinstance(msg, AIMessage) and i == len(st.session_state.messages) - 1:
            display_feedback_buttons(i)

def display_feedback_buttons(message_index):
    if message_index not in st.session_state.feedback:
        col1, col2 = st.columns([0.1, 0.1])
        with col1:
            if st.button("👍", key=f"thumbs_up_{message_index}"):
                st.session_state.feedback[message_index] = "positive"
                # Log feedback without reprocessing conversation
                log_feedback_data(st.session_state.feedback, st.session_state.messages)
                st.rerun()
        with col2:
            if st.button("👎", key=f"thumbs_down_{message_index}"):
                st.session_state.feedback[message_index] = "negative"
                # Log feedback without reprocessing conversation
                log_feedback_data(st.session_state.feedback, st.session_state.messages)
                st.rerun()

def process_user_input(prompt):
    st.session_state.messages.append(HumanMessage(content=prompt))
    config = {"recursion_limit": 100}  # Feedback is not needed in the config now.
    try:
        response = st.session_state.workflow.invoke(
            {"messages": st.session_state.messages},
            config
        )
    except Exception as e:
        st.error(f"An error occurred while processing the request: {e}")
        return

    if "messages" in response:
        st.session_state.messages = response["messages"]
    else:
        st.error("Unexpected workflow response format.")

# Main App Layout
st.title("LangGraph Chatbot with Feedback")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow" not in st.session_state:
    st.session_state.workflow = create_workflow()
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

display_chat_history()

user_prompt = st.chat_input("Type your message")
if user_prompt:
    process_user_input(user_prompt)
    st.rerun()
