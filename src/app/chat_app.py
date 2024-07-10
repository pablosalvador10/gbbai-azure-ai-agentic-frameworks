import streamlit as st
import asyncio
from dotenv import load_dotenv
import os
from autogen import ConversableAgent, GroupChat, GroupChatManager
from typing import List, Dict, Any

# Function to initialize llm_config
def initialize_llm_config():
    global llm_config  # Declare llm_config as global to modify it
    if 'llm_config' not in globals():  # Check if llm_config is not already defined
        # Load environment variables from a .env file
        load_dotenv()

        # Azure Open AI Completion Configuration
        AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
        AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID = os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
        AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

        llm_config = {
            "config_list": [
                {
                    "model": AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID,
                    "api_type": "azure",
                    "api_key": AZURE_OPENAI_KEY,
                    "base_url": AZURE_OPENAI_API_ENDPOINT,
                    "api_version": AZURE_OPENAI_API_VERSION
                }
            ]
        }

    return llm_config

if "llm_initiated" not in st.session_state:
    st.session_state.llm_initiated = initialize_llm_config()  

# Set up page title and description
st.set_page_config(page_title="Medical Research Chat App", page_icon="🤖", layout="wide")

st.markdown(
    "This is a demo of AutoGen chat agents designed for medical research. You can use it to chat with OpenAI's GPT-4 models. "
    "They are able to execute commands, answer questions, and even write research documents."
)

# Trackable agents for displaying messages in Streamlit
class TracableConversableAgent(ConversableAgent):
    def _process_received_message(self, message, sender, silent):
        if 'function_call' not in message:
            with st.chat_message(sender.name, avatar="👤"):
                st.markdown(message['content'])
        return super()._process_received_message(message, sender, silent)

# Define the agents
medical_research_planner = TracableConversableAgent(
    name="MedicalResearchPlanner",
    system_message= "Given a research task, your role is to determine the specific information needed to comprehensively support the research. "
    "You will assess the task's progress and delegate sub-tasks to other agents as needed. ",
    llm_config=st.session_state.llm_initiated,
)

final_medical_reviewer = TracableConversableAgent(
    name="FinalMedicalReviewer",
    system_message="You are the final medical reviewer, tasked with aggregating and reviewing feedback from other reviewers. "
                  "Your role is to make the final decision on the content's readiness for publication, ensuring it adheres to all legal, "
                  "security, and ethical standards. If documentation is reduced to public circulation, mention STOP DOCUMENT LOOKS GOOD.",
    llm_config=st.session_state.llm_initiated,
)

medical_researcher = TracableConversableAgent(
    name="MedicalResearcher",
    system_message="As a Medical Researcher, your role is to draft a comprehensive manuscript detailing your study's findings. "
                   "Ensure the manuscript is scientifically robust, covering all critical aspects of your research.",
    llm_config=st.session_state.llm_initiated,
)

agents_dict = {
    "MedicalResearchPlanner": medical_research_planner,
    "FinalMedicalReviewer": final_medical_reviewer,
    "MedicalResearcher": medical_researcher
}

# Define the group chat for the medical use case
medical_groupchat = GroupChat(
    agents=[
        agents_dict["MedicalResearchPlanner"], agents_dict["FinalMedicalReviewer"], 
        agents_dict["MedicalResearcher"]
    ],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        agents_dict["MedicalResearchPlanner"]: [
            agents_dict["FinalMedicalReviewer"], agents_dict["MedicalResearcher"]
        ],
        agents_dict["FinalMedicalReviewer"]: [
            agents_dict["MedicalResearcher"]
        ],
        agents_dict["MedicalResearcher"]: [
            agents_dict["FinalMedicalReviewer"]
        ]
    },
    speaker_transitions_type="allowed",
)

# Initialize the GroupChatManager with the medical group chat and LLM configuration
medical_manager = GroupChatManager(
    groupchat=medical_groupchat, llm_config=st.session_state.llm_initiated
)

# Setup main area: user input and chat messages
with st.container():
    # Assuming st.session_state.chat_history is initialized somewhere in your code
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_input = st.text_input("User Input", placeholder="Type your message here...")
    
    # Check if the current input has already been processed
    if user_input and user_input not in st.session_state.chat_history:
        # Add the current input to the chat history to avoid repetition
        st.session_state.chat_history.append(user_input)
    
        # Create an event loop: this is needed to run asynchronous functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
        async def initiate_chat():
            with st.container():
                st.markdown(f"**User**: {user_input}")
    
            # Define a custom termination check function
            def is_termination_msg_from_final_reviewer(message, sender):
                # Check if the sender is the FinalMedicalReviewer and the message contains the termination phrase, ignoring case
                return sender.name == "FinalMedicalReviewer" and "stop document looks good" in message.get("content", "").lower()
    
            # Initiate the chat with the custom termination check
            result = await medical_research_planner.a_initiate_chat(
                recipient=medical_manager,
                message=user_input,
                max_turns=5,
                is_termination_msg=is_termination_msg_from_final_reviewer
            )
    
            # If the chat is terminated, display the final approval message
            if result.get("terminated", False):
                with st.container():
                    st.markdown("**Final document is being approved**")
            else:
                st.stop()  # Stop code execution if not terminated
    
        # Run the asynchronous function within the event loop
        loop.run_until_complete(initiate_chat())
    
        # Close the event loop
        loop.close()