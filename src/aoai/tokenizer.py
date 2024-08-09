"""
`tokenizer.py` is a module that extends the AzureOpenAIManager to include tokenization capabilities for Azure OpenAI.
"""

from typing import Dict, List, Optional, Union

import tiktoken
# Load environment variables from .env file
from dotenv import load_dotenv

from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

load_dotenv()


class AzureOpenAITokenizer:
    """
    This class is a tokenizer for Azure OpenAI. It provides methods to call the Azure OpenAI API
    and extract rate limit and usage information from the response. It does not extend any other class.
    """

    DEFAULT_MODEL = "gpt-4-32k-0314"
    TOKENS_PER_MESSAGE = {
        "gpt-3.5-turbo-0613": 3,
        "gpt-3.5-turbo-16k-0613": 3,
        "gpt-4-0314": 3,
        "gpt-4-32k-0314": 3,
        "gpt-4-0613": 3,
        "gpt-4-32k-0613": 3,
        "gpt-3.5-turbo-0301": 4,
    }
    TOKENS_PER_NAME = {
        "gpt-3.5-turbo-0301": -1,
    }

    def __init__(self, model: Optional[str] = None):
        """
        Initialize the AzureOpenAITokenizer class with an optional model.

        :param model: The name of the model to use for token estimation. If not provided, defaults to DEFAULT_MODEL.
        """
        self.model = model if model is not None else self.DEFAULT_MODEL

    def estimate_tokens_azure_openai(
        self,
        messages: List[Dict[str, Union[str, int]]],
        model: Optional[str] = None,
        has_function_call: bool = False,
    ) -> int:
        """
        Estimates the number of tokens used by a list of messages for a specific OpenAI model.

        This function estimates the token count for a given set of messages based on the model's specific encoding and formatting rules.

        :param messages (List[Dict[str, Union[str, int]]]): A list of messages, each represented as a dictionary.
        :param model (str): The model name, which determines the encoding and token counting rules. Default is "gpt-3.5-turbo-0613".
        :param has_function_call (bool): Flag to indicate if there is a function call in the messages, which affects token count.

        :return (int): The estimated number of tokens for the provided messages.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = self.TOKENS_PER_MESSAGE.get(model, 3)
        tokens_per_name = self.TOKENS_PER_NAME.get(model, 1)

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key in ["role", "content", "name"] and isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

        if has_function_call:
            num_tokens += 9

        num_tokens += 3

        return num_tokens

    def estimate_tokens_completion(
        self,
        response: str,
        model: Optional[str] = None,
    ) -> int:
        """
        Estimates the number of tokens used by a text string for a specific OpenAI model.

        This function estimates the token count for a given text based on the model's specific encoding rules.

        :param text (str): The text string to estimate tokens for.
        :param model (str): The model name, which determines the encoding and token counting rules. Default is "gpt-3.5-turbo-0613".

        :return (int): The estimated number of tokens for the provided text.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = len(encoding.encode(response))

        return num_tokens
