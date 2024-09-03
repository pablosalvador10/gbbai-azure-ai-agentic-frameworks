"""agentcenter.py"""

import streamlit as st
from src.app.autogenhelper import StreamlitConversableAgent, get_llm_config
from typing import Dict

def initialize_default_agents() -> None:
    agents_loaded = False  

    llm_config = get_llm_config()

    medical_research_planner = StreamlitConversableAgent(
        name="MedicalResearchPlanner",
        system_message=(
            "Given a research task, your role is to determine the specific information needed to comprehensively support the research. "
            "You will assess the task's progress and delegate sub-tasks to other agents as needed. If no more information is needed and document looks good, mention STOP or TERMINATE"
        ),
        llm_config=llm_config,
        avatar="🧑🏿‍💼",
        verbose=True
    )

    final_medical_reviewer = StreamlitConversableAgent(
        name="FinalMedicalReviewer",
        system_message=(
            "You are the final medical reviewer, tasked with aggregating and reviewing feedback from other reviewers. "
            "Your role is to make the final decision on the content's readiness for publication, ensuring it adheres to all legal, "
            "security, and ethical standards. If documentation is ready for public circulation, mention STOP or TERMINATE"
        ),
        llm_config=llm_config,
        avatar="👨🏽‍⚕️",
    )

    medical_researcher = StreamlitConversableAgent(
        name="MedicalResearcher",
        system_message=(
            "As a Medical Researcher, your role is to draft a comprehensive manuscript detailing your study's findings. "
            "Ensure the manuscript is scientifically robust, covering all critical aspects of your research."
        ),
        llm_config=llm_config,
        avatar="👩‍⚕️",
    )

    if 'agents' not in st.session_state:
        st.session_state.agents = {}

    st.session_state.agents['MedicalResearchPlanner'] = {
        'object': medical_research_planner,
        'name': medical_research_planner.name,
        'avatar': medical_research_planner.avatar,
        'llm_config': llm_config,
        'system_message': medical_research_planner.system_message
    }
    st.session_state.agents['FinalMedicalReviewer'] = {
        'object': final_medical_reviewer,
        'name': final_medical_reviewer.name,
        'avatar': final_medical_reviewer.avatar,
        'llm_config': llm_config,
        'system_message': final_medical_reviewer.system_message
    }
    st.session_state.agents['MedicalResearcher'] = {
        'object': medical_researcher,
        'name': medical_researcher.name,
        'avatar': medical_researcher.avatar,
        'llm_config': llm_config,
        'system_message': medical_researcher.system_message
    }

    if ('MedicalResearchPlanner' in st.session_state.agents and
        'FinalMedicalReviewer' in st.session_state.agents and
        'MedicalResearcher' in st.session_state.agents):
        agents_loaded = True

    st.session_state.agents_loaded = agents_loaded

def update_agent(agent_name: str, 
                 updated_avatar: str, 
                 updated_system_message: str, 
                 updated_llm_config: Dict) -> None:

    # Create a new agent object with the updated details
    updated_agent = StreamlitConversableAgent(
        name=agent_name,
        system_message=updated_system_message,
        llm_config=updated_llm_config,
        avatar=updated_avatar,
    )
    st.session_state.agents[agent_name]['avatar'] = updated_avatar
    st.session_state.agents[agent_name]['system_message'] = updated_system_message
    st.session_state.agents[agent_name]['llm_config'] = updated_llm_config
    st.session_state.agents[agent_name]['object'] = updated_agent

    st.experimental_rerun()

def display_agents() -> None:
    """
    Display and update the parameters of the agents.

    This function renders forms for each agent in the Streamlit sidebar, allowing users to view and modify agent parameters.
    """
    if 'agents' in st.session_state:
        agent_tabs = st.sidebar.tabs([f"{info['avatar']} {name}" for name, info in st.session_state.agents.items()])
        for tab, (agent_name, agent_info) in zip(agent_tabs, st.session_state.agents.items()):
            agent = agent_info['object']
            with tab:
                with st.form(key=f"form_{agent_name}"):   
                    updated_system_message = st.text_area("📝 Task or Expertise", value=agent_info['system_message'], help="Describe the task or expertise of the agent. This helps in defining the agent's role and capabilities.")

                    # Get available deployments from session state
                    if "deployments" in st.session_state and st.session_state.deployments:
                        available_deployments = list(st.session_state.deployments.keys())
                        selected_deployment_name = st.radio(
                            "🧠 Choose your agent brain",
                            options=available_deployments,
                            index=available_deployments.index(agent_info['llm_config']) if agent_info['llm_config'] in available_deployments else 0,
                            help= "The deployment could be an LLM (Large Language Model) or SLM (Small Language Model). Choose the appropriate model for your agent."
                        )

                        # Additional inputs for the selected deployment
                        deployment_key = st.session_state.deployments[selected_deployment_name]['key']
                        deployment_endpoint = st.session_state.deployments[selected_deployment_name]['endpoint']
                        deployment_version = st.session_state.deployments[selected_deployment_name]['version']

                        # Generate the LLM config
                        updated_llm_config = get_llm_config(
                            azure_openai_key=deployment_key,
                            azure_aoai_chat_model_name_deployment_id=selected_deployment_name,
                            azure_openai_api_endpoint=deployment_endpoint,
                            azure_openai_api_version=deployment_version
                        )
                    else:
                        st.error("No deployments available. Please add a deployment in the Deployment Center.")
                    # Define a list of emoji options for the avatar
                    # emoji_options = ["👩‍⚕️", "👩🏿‍⚕️", "👩‍⚕️"]
                    # if agent_info['avatar'] not in emoji_options:
                    #     emoji_options.append(agent_info['avatar'])
                    
                    # updated_avatar = st.radio("Avatar", options=emoji_options, index=emoji_options.index(agent_info['avatar']))

                    submitted = st.form_submit_button("Update")
                    updated_avatar = agent_info['avatar']
                    if submitted:
                        update_agent(agent_name, updated_avatar, updated_system_message, updated_llm_config)



def create_agent_center() -> None: 
    st.markdown("## 🛠️ Agent Center")
    initialize_default_agents()
    display_agents()
