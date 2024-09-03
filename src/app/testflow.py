import logging
from PIL import Image
import argparse
from autogen import register_function, ConversableAgent, AssistantAgent, runtime_logging
from autogen.agentchat.contrib import img_utils
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file
load_dotenv()

# import tools 
from src.tools.inpainting import edit_image, inpainting, sam, refiner, generate_tags_and_boxes

# Azure Open AI Completion Configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID = os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

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



def extract_images(sender: ConversableAgent, recipient: ConversableAgent) -> Image:
    """
    Extracts images from the conversation history between two agents.
    """
    images = []
    all_messages = sender.chat_messages[recipient]

    for message in reversed(all_messages):
        contents = message.get("content", [])
        for content in contents:
            if isinstance(content, str):
                continue
            if content.get("type", "") == "image_url":
                img_data = content["image_url"]["url"]
                images.append(img_utils.get_pil_image(img_data))

    if not images:
        raise ValueError("No image data found in messages.")

    return images

# Initialize Agents
planner_agent = ConversableAgent(
    name="Planner",
    system_message=(
        "As a Planner, your role is to create a comprehensive plan for editing the provided image based on the user's prompt. "
        "You will outline the steps and tools needed to achieve the desired result. Your plan should be detailed and logical, "
        "focusing on aspects such as composition, color balance, and any specific elements mentioned in the user's prompt. "
        "You will provide a step-by-step guide for the Image Editor to follow, ensuring that all necessary edits are clearly defined."
    ),
    llm_config=llm_config,
)

image_editor_agent = AssistantAgent(
    name="Image Editor",
    system_message=(
        "You are an Image Editor. Your role is to apply the suggested edits to the image based on the "
        "Planner's analysis and the user's preferences. You will use various tools and techniques "
        "to modify the image, ensuring that the final result meets the user's expectations. Your goal is "
        "to produce a polished and visually appealing image. "
        "You have access to the following tools:\n"
        "- inpainting: Perform inpainting on an image based on the provided prompt.\n"
        "- refiner: Refine an image based on the provided prompt.\n"
        "- generate_tags_and_boxes: Generate tags and bounding boxes for an image.\n"
        "- sam: Run SAM on an image with the provided bounding boxes.\n"
        "- edit_image: Edit an image using object detection, segmentation, and inpainting techniques."
    ),
    llm_config=llm_config,
)

user_proxy = ConversableAgent(
    name="User Proxy",
    llm_config=None,  # Replace with your actual LLM configuration
    is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
    human_input_mode="NEVER",
)

for caller in [planner_agent, image_editor_agent]:
    register_function(
        inpainting,
        caller=caller,
        executor=user_proxy,
        name="inpainting",
        description="Perform inpainting on an image based on the provided prompt."
    )
    register_function(
        refiner,
        caller=caller,
        executor=user_proxy,
        name="refiner",
        description="Refine an image based on the provided prompt."
    )
    register_function(
        generate_tags_and_boxes,
        caller=caller,
        executor=user_proxy,
        name="generate_tags_and_boxes",
        description="Generate tags and bounding boxes for an image."
    )
    register_function(
        sam,
        caller=caller,
        executor=user_proxy,
        name="sam",
        description="Run SAM on an image with the provided bounding boxes."
    )
    register_function(
        edit_image,
        caller=caller,
        executor=user_proxy,
        name="edit_image",
        description="Edit an image using object detection, segmentation, and inpainting techniques."
    )

def image_editing_session(image_path: str, initial_prompt: str):
    """
    Conducts an image editing session based on the user's prompt.
    """
    while True:
        message_content = f"Image Path: {image_path}\nPrompt: {initial_prompt}"

        chat_result = planner_agent.initiate_chat(
            recipient=image_editor_agent,
            message={"content": message_content},
            max_turns=5,
        )

        # Extract the edited image from the chat result
        edited_images = extract_images(image_editor_agent, planner_agent)

        feedback = input("Are you happy with the image? (yes/no): ")
        
        if feedback.lower() == 'yes':
            logging.info("Image editing complete.")
            break
        else:
            initial_prompt = input("Please describe the changes you want: ")

def main():
    """
    Main entry point for the image editing script.
    """
    parser = argparse.ArgumentParser(description="Image Editing Session")
    parser.add_argument('--image_path', type=str, required=True, help="Path or URL to the image")
    parser.add_argument('--prompt', type=str, required=True, help="Initial prompt for editing the image")

    args = parser.parse_args()

    image_editing_session(args.image_path, args.prompt)

if __name__ == "__main__":
    logging_session_id = runtime_logging.start(config={"dbname": "logs.db"})
    logging.basicConfig(level=logging.INFO)
    print(f"Logging session started with ID: {logging_session_id}")
    main()
    runtime_logging.stop()
