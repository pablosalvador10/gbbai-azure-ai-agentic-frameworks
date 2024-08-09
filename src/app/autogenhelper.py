from autogen import ConversableAgent, GroupChat, GroupChatManager
import yaml
from typing import Dict, List, Optional, Union, Callable, Literal, Any
import os
import streamlit as st

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

}

initialize_session_state(session_vars, initial_values)

def get_llm_config() -> Dict[str, List[Dict[str, str]]]:
    """
    Generate a configuration list dictionary for the LLM from environment variables.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing the configuration list.
    """
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_aoai_chat_model_name_deployment_id = os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
    azure_openai_api_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    return {
        "config_list": [
            {
                "model": azure_aoai_chat_model_name_deployment_id,
                "api_type": "azure",
                "api_key": azure_openai_key,
                "base_url": azure_openai_api_endpoint,
                "api_version": azure_openai_api_version
            }
        ]
    }

from typing import Optional, Union, Callable, Dict, List
from typing_extensions import Literal
import streamlit as st

class SuperConversableAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        chat_messages: Optional[Dict[ConversableAgent, List[Dict]]] = None,
        avatar: str = '🤖',
        silent: bool = False
    ):
        """
        Initialize the SuperConversableAgent with LLM configuration.

        Args:
            name (str): The name of the agent.
            system_message (str or list, optional): The system message for the agent. Defaults to a helpful assistant message.
            is_termination_msg (callable, optional): Function to check for termination message.
            max_consecutive_auto_reply (int, optional): Maximum number of consecutive auto replies.
            human_input_mode (str, optional): Human input mode. Defaults to "TERMINATE".
            function_map (dict, optional): Mapping of function names to callables.
            code_execution_config (dict or bool, optional): Code execution configuration. Defaults to False.
            llm_config (dict or bool, optional): LLM configuration. Defaults to None.
            default_auto_reply (str or dict, optional): Default auto reply. Defaults to empty string.
            description (str, optional): Description of the agent.
            chat_messages (dict, optional): Initial chat messages for the agent.
            avatar (str, optional): The avatar for the agent. Defaults to a robot emoji.
            silent (bool, optional): Whether the agent should process messages silently. Defaults to False.
        """
        self.avatar = avatar
        self.silent = silent
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages
        )

    def _process_received_message(self, message, sender, silent=None):
        """
        Process received message and log it in Streamlit chat interface.
    
        Args:
            message (str): The message content.
            sender (ConversableAgent): The sender of the message.
            silent (bool, optional): Whether the message should be processed silently. Defaults to the instance's silent attribute.
        """
        if silent is None:
            silent = self.silent
        if not silent:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({'role': sender.name, 'content': message})
    
        return super()._process_received_message(message, sender, silent)

    @classmethod
    def load_agent(cls, yaml_file: str) -> 'SuperConversableAgent':
        """
        Load an agent configuration from a YAML file.

        Args:
            yaml_file (str): Path to the YAML configuration file.

        Returns:
            TracableConversableAgent: An instance of TracableConversableAgent.
        """
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        llm_config = get_llm_config() if config.get('llm_config') == "default" else config.get('llm_config')

        return cls(
            name=config['name'],
            system_message=config.get('system_message', "You are a helpful AI Assistant."),
            is_termination_msg=config.get('is_termination_msg'),
            max_consecutive_auto_reply=config.get('max_consecutive_auto_reply'),
            human_input_mode=config.get('human_input_mode', "TERMINATE"),
            function_map=config.get('function_map'),
            code_execution_config=config.get('code_execution_config', False),
            llm_config=llm_config,
            default_auto_reply=config.get('default_auto_reply', ""),
            description=config.get('description'),
            chat_messages=config.get('chat_messages'),
            avatar=config.get('avatar', '🤖')
        )
    
