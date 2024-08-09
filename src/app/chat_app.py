import streamlit as st
import asyncio
from autogen import GroupChat, GroupChatManager
from src.app.medicalAgents import medical_research_planner, final_medical_reviewer, medical_researcher
from src.app.autogenhelper import get_llm_config
from typing import List, Dict, Any
import autogen
import re
from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

def initialize_session_state(vars: List[str], initial_values: Dict[str, Any]) -> None:
    """
    Initialize Streamlit session state with default values if not already set.

    :param vars: List of session state variable names.
    :param initial_values: Dictionary of initial values for the session state variables.
    """
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = initial_values.get(var, None)


session_vars = [
    "conversation_history",
    "chat_history",
    "messages",
    "agents_loaded",
]
initial_values = {
    "conversation_history": [],
    "chat_history": [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        }
    ],
    "messages": [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        },
    ],
    "agents_loaded": False

}

initialize_session_state(session_vars, initial_values)
    
from utils.ml_logging import get_logger
logger = get_logger()
 
def initialize_agents():
    agents_loaded = False  # Initialize the variable to False

    if 'MedicalResearchPlanner' not in st.session_state:
        st.session_state.MedicalResearchPlanner = medical_research_planner
    if 'FinalMedicalReviewer' not in st.session_state:
        st.session_state.FinalMedicalReviewer = final_medical_reviewer
    if 'MedicalResearcher' not in st.session_state:
        st.session_state.MedicalResearcher = medical_researcher

    if ('MedicalResearchPlanner' in st.session_state and
        'FinalMedicalReviewer' in st.session_state and
        'MedicalResearcher' in st.session_state):
        agents_loaded = True

    st.session_state.agents_loaded = agents_loaded

initialize_agents()

st.set_page_config(page_title="Medical Research Chat App", page_icon="🤖", layout="wide")

agents_dict = {
    "MedicalResearchPlanner": st.session_state.MedicalResearcher,
    "FinalMedicalReviewer": st.session_state.FinalMedicalReviewer,
    "MedicalResearcher": st.session_state.MedicalResearcher
}

medical_groupchat = GroupChat(
    agents=[
        agents_dict["MedicalResearchPlanner"], 
        agents_dict["FinalMedicalReviewer"], 
        agents_dict["MedicalResearcher"]
    ],
    messages=[],
    max_round=3,
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

llm_config = get_llm_config()

medical_manager = GroupChatManager(
    groupchat=medical_groupchat, llm_config=llm_config
)

def initialize_chatbot() -> None:
    """
    Initialize a chatbot interface for user interaction with enhanced features.
    """
    st.markdown(
        "<h4 style='text-align: center;'>AgentBuddy 🤖</h4>",
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        }
    ]
        
    respond_conatiner = st.container(height=400)

    with respond_conatiner:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            avatar_style = "🧑‍💻" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar_style):
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px;'>{content}</div>",
                    unsafe_allow_html=True,
                )
    warning_issue_quality = st.empty()
    if st.session_state.get("agents_loaded") is False:
        warning_issue_quality.warning(
            "Oops! It seems I'm currently unavailable. 😴 Please ensure the LLM is configured correctly in the Benchmark Center and Buddy settings. Need help? Refer to the 'How To' guide for detailed instructions! 🧙"
        )

    prompt = st.chat_input("Ask away!")
    if prompt:
        with respond_conatiner:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message('human', avatar='🧑‍💻'):
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px;'>{prompt}</div>",
                    unsafe_allow_html=True,
                )
            logger.info(f"User input: {prompt}")
            terminate_pattern = re.compile(r".*STOP.*", re.IGNORECASE)
            terminate_variation_pattern = re.compile(r".*TERMINATED.*", re.IGNORECASE)
            finalize_pattern = re.compile(r".*TERMINATE.*", re.IGNORECASE)

            is_termination_msg = lambda x: any([
                terminate_pattern.search(x.get("content", "")),
                terminate_variation_pattern.search(x.get("content", "")),
                finalize_pattern.search(x.get("content", "")),
            ])

            # Start logging with logger_type and the filename to log to
            logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "runtime.log"})
            print("Logging session ID: " + str(logging_session_id))

            result = medical_research_planner.initiate_chat(
                recipient=medical_manager,
                message=prompt,
                max_turns=2,
                is_termination_msg=is_termination_msg
            )
            autogen.runtime_logging.stop()
    


def main():
    """
    Main function to run the chatbot interface.
    """
    st.markdown(
    "This is a demo of AutoGen chat agents designed for medical research. You can use it to chat with OpenAI's GPT-4 models. "
    "They are able to execute commands, answer questions, and even write research documents.")
    initialize_chatbot()
    

if __name__ == "__main__":
    main()
 