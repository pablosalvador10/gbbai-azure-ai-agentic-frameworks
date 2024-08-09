import streamlit as st
import asyncio
from autogen import GroupChat, GroupChatManager
from src.app.autogenhelper import get_llm_config
from typing import List, Dict, Any, Optional
import autogen
from src.app.autogenhelper import SuperConversableAgent, get_llm_config
from src.app.deploymentcenter import load_default_deployment, create_benchmark_center, display_deployments
from src.app.agentscenter import create_agent_center
import re
from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

st.set_page_config(page_title="Medical Research Chat App", page_icon="🤖", layout="centered")

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
    "deployments",
    "settings_agent"
]
initial_values = {
    "conversation_history": [],
    "chat_history": [
        {
            "role": "assistant",
            "content": (
                "🚀 Hi there! "
                "Please ask me a question, and I can look up sources like PubMed to give you the best answer. "
                "I'm also happy to write drafts of research documents as needed. 😄"
            ),
        }
    ],
    "messages": [
        {
            "role": "assistant",
            "content": (
                "🚀 Hi there! "
                "Please ask me a question, and I can look up sources like PubMed to give you the best answer. "
                "I'm also happy to write drafts of research documents as needed. 😄"
            ),
        }
    ],
    "agents_loaded": False,
    "deployments": {},
}

initialize_session_state(session_vars, initial_values)

def initialize_groupchat():
    agents_dict = {
        "MedicalResearchPlanner": st.session_state.agents['MedicalResearchPlanner']['object'],
        "FinalMedicalReviewer": st.session_state.agents['FinalMedicalReviewer']['object'],
        "MedicalResearcher": st.session_state.agents['MedicalResearcher']['object']
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

    if "medical_manager" not in st.session_state:
        st.session_state.MedicalManager = medical_manager

def configure_agent_settings(agent_name: str) -> dict:
    """
    Configure model settings for an agent and return the values from each input.

    :return: A dictionary containing the settings values.
    """
    with st.expander(f"{agent_name} Settings", expanded=False):
        if f"settings_{agent_name}" not in st.session_state:
            st.session_state[f"settings_{agent_name}"] = {}

        st.session_state[f"settings_{agent_name}"]["model"] = st.selectbox(
            f"Model for {agent_name}",
            options=["Model A", "Model B", "Model C"],
            help="Select the model to be used by this agent."
        )
        st.session_state[f"settings_{agent_name}"]["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Think of 'Temperature' as the chatbot's creativity control. A higher value makes the chatbot more adventurous, giving you more varied and surprising responses.",
        )
        st.session_state[f"settings_{agent_name}"]["top_p"] = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="The 'Top P' setting helps fine-tune how the chatbot picks its words. Lowering this makes the chatbot's choices more predictable and focused, while a higher setting encourages diversity in responses.",
        )
        st.session_state[f"settings_{agent_name}"]["max_tokens"] = st.slider(
            "Max Generation Tokens (Input)",
            min_value=100,
            max_value=3000,
            value=1000,
            help="This slider controls how much the chatbot talks. Slide to the right for longer stories or explanations, and to the left for shorter, more concise answers.",
        )
        st.session_state[f"settings_{agent_name}"]["presence_penalty"] = st.slider(
            "Presence Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Use 'Presence Penalty' to manage repetition. Increase it to make the chatbot avoid repeating itself, making each response fresh and unique.",
        )
        st.session_state[f"settings_{agent_name}"]["frequency_penalty"] = st.slider(
            "Frequency Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="The 'Frequency Penalty' discourages the chatbot from using the same words too often. A higher penalty encourages a richer vocabulary in responses.",
        )

    return st.session_state[f"settings_{agent_name}"]

def configure_sidebar() -> None:
    """
    Configure the sidebar with benchmark Center and deployment forms, allowing users to choose between evaluating a Large Language Model (LLM) or a System based on LLM.
    """
    with st.sidebar:
        st.markdown("## 🤖 Deployment Center")
        if st.session_state.deployments == {}:
            load_default_deployment()
        create_benchmark_center()
        display_deployments()

        st.sidebar.divider()
        
        create_agent_center()

def initialize_chatbot() -> None:
    """
    Initialize a chatbot interface for user interaction with enhanced features.
    """
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
    if "messages_quality" not in st.session_state:
        st.session_state.messages_quality = [
        {
            "role": "assistant",
            "content": (
                "🚀 Ask away! I am all ears and ready to dive into your queries. "
                "I'm here to make sense of the numbers from your benchmarks and support you during your analysis! 😄📊"
            ),
        },
    ]
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        avatar_style = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar_style):
            st.markdown(
                f"<div style='padding: 10px; border-radius: 5px;'>{content}</div>",
                unsafe_allow_html=True,
            )


async def main():

    configure_sidebar()
    st.markdown(
        "<h4 style='text-align: center;'> Medical Research Assistant 🤖</h4>",
        unsafe_allow_html=True,
    )
    chat_container = st.container(height=500)
    initialize_groupchat()
    with chat_container:
        initialize_chatbot()
    prompt = st.chat_input("Ask away!")
    if prompt:
        with chat_container:
            with st.chat_message('You', avatar='👩🏻'):
                formatted_message = f"""
                <div style='padding: 10px; border-radius: 5px;'>
                    <strong>User</strong>
                    <p>{prompt}</p>
                </div>
                """
                st.markdown(formatted_message, unsafe_allow_html=True)
            logger.info(f"User input: {prompt}")

            terminate_pattern = re.compile(r".*STOP.*", re.IGNORECASE)
            terminate_variation_pattern = re.compile(r".*TERMINATED.*", re.IGNORECASE)
            finalize_pattern = re.compile(r".*TERMINATE.*", re.IGNORECASE)

            is_termination_msg = lambda x: any([
                terminate_pattern.search(x.get("content", "")),
                terminate_variation_pattern.search(x.get("content", "")),
                finalize_pattern.search(x.get("content", "")),
            ])

            logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "runtime.log"})
            print("Logging session ID: " + str(logging_session_id))

            chat_result = await st.session_state.agents['MedicalResearcher']['object'].a_initiate_chat(
                recipient=st.session_state.MedicalManager,
                message=prompt,
                max_turns=2,
                is_termination_msg=is_termination_msg
            )
            autogen.runtime_logging.stop()
           
            logger.info(f"Final Response: {chat_result.chat_history[-1]['content']}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
