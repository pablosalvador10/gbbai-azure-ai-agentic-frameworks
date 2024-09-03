import os
import ssl
import json
import requests
import base64
import traceback
import logging
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Any
from utils.ml_logging import get_logger

logger = get_logger()

def allow_self_signed_https(allowed: bool) -> None:
    """
    Allows bypassing SSL certificate verification, particularly useful in environments where
    self-signed certificates are in use. This function ensures connections proceed smoothly
    even if certificates aren't from a trusted authority.

    :param allowed: Set to True to bypass verification, False to enforce it.
    :return: None
    :raises: None
    """
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def make_request(url: str, body: dict, api_key: str) -> Optional[bytes]:
    """
    Handles making POST requests to an API with JSON data and returns the response content.

    :param url: The API endpoint URL.
    :param body: The request payload as a dictionary.
    :param api_key: The API key for authorization.
    :return: The API response content as bytes, or None if the request fails.
    :raises requests.exceptions.HTTPError: If there is an HTTP error during the request.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    try:
        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.content
    except requests.exceptions.HTTPError as error:
        logging.error(f"Request failed: {error}")
        return None


def inpainting(image: str, mask_image: str, prompt: str) -> Optional[bytes]:
    """
    Performs inpainting on an image using Azure ML by filling in the masked areas based on a text prompt.

    :param image: Base64 encoded image string.
    :param mask_image: Base64 encoded mask image string indicating areas to modify.
    :param prompt: Text prompt describing the desired inpainting result.
    :return: The processed image data returned by the Azure service, or None on failure.
    :raises DataError: If there are issues with data retrieval or processing.
    """
    allow_self_signed_https(True)
    data = {
        "input_data": {
            "data": {
                "prompt": prompt,
                "image": image,
                "mask_image": mask_image,
                "negative_prompt": "blurry,cartoonish,dog"
            },
            "columns": ["prompt", "image", "mask_image"],
            "index": [0],
            "parameters": {
                "num_inference_steps": 500,
                "guidance_scale": 7.5
            }
        }
    }

    url = os.getenv('INPAINTING_URL')
    api_key = os.getenv('INPAINTING_API_KEY')
    return make_request(url, data, api_key)


def refiner(image: str, prompt: str) -> Optional[bytes]:
    """
    Refines an image by improving its quality or modifying it based on a given prompt.

    :param image: Base64 encoded image string to be refined.
    :param prompt: Text prompt guiding the refinement process.
    :return: The refined image data, or None if the request fails.
    :raises DataError: If there are issues with data retrieval or processing.
    """
    allow_self_signed_https(True)
    data = {
        "input_data": {
            "data": [{"prompt": prompt, "image": image}],
            "columns": ["prompt", "image"],
            "index": [0]
        }
    }

    url = os.getenv('REFINER_URL')
    api_key = os.getenv('REFINER_API_KEY')
    return make_request(url, data, api_key)


def base64_to_image(base64_string: str) -> bytes:
    """
    Converts a Base64 encoded string into image bytes, ready for further processing or visualization.

    :param base64_string: The Base64 encoded string of the image.
    :return: The decoded image bytes.
    :raises ValueError: If there is an error decoding the base64 string.
    """
    try:
        if "data:image" in base64_string:
            base64_string = base64_string.split(",")[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        logging.error(f"Error decoding base64 string: {e}")
        raise ValueError("Failed to decode base64 string")


def create_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Creates a PIL Image object from raw image bytes, enabling further image manipulation or saving the image to disk.

    :param image_bytes: The raw bytes of the image.
    :return: The PIL Image object created from the bytes.
    :raises ValueError: If there is an error creating the image from bytes.
    """
    try:
        image_stream = BytesIO(image_bytes)
        return Image.open(image_stream)
    except Exception as e:
        logging.error(f"Error creating image from bytes: {e}")
        raise ValueError("Failed to create image from bytes")


def image_to_base64(image_path_or_url: str) -> Optional[str]:
    """
    Converts an image from a file path or URL to a Base64 encoded string.

    :param image_path_or_url: The local path or URL of the image to encode.
    :return: The Base64 encoded string of the image, or None if an error occurs.
    :raises ValueError: If there is an error converting the image to base64.
    """
    try:
        if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        logger.error(traceback.format_exc())
        raise


def generate_tags_and_boxes(image_path: str) -> List[Dict[str, Any]]:
    """
    Generates tags and bounding boxes for objects detected in an image using Azure's Computer Vision service.

    :param image_path: The path or URL of the image to analyze.
    :return: A list of dictionaries, each containing bounding box coordinates, confidence score, and the object tag.
    :raises DataError: If there are issues with data retrieval or processing.
    """
    try:
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.ai.vision.imageanalysis.models import VisualFeatures
        from azure.core.credentials import AzureKeyCredential

        endpoint = os.getenv("VISION_ENDPOINT")
        key = os.getenv("VISION_KEY")

        if not endpoint or not key:
            logger.error("Azure Computer Vision endpoint or key is not set.")
            return []

        client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        logger.info(f"Initialized ImageAnalysisClient with endpoint: {endpoint}")

        if image_path.startswith(('http://', 'https://')):
            logger.info(f"Analyzing image from URL: {image_path}")
            result = client.analyze_from_url(
                image_url=image_path,
                visual_features=[VisualFeatures.OBJECTS, VisualFeatures.TAGS],
                gender_neutral_caption=False,
            )
        else:
            logger.info(f"Analyzing image from local path: {image_path}")
            with open(image_path, "rb") as f:
                image_data = f.read()
            result = client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.OBJECTS, VisualFeatures.TAGS],
                gender_neutral_caption=False,
            )

        logger.info("Image analysis completed.")
        objs = []
        if 'values' in result.objects:
            logger.info(f"Detected {len(result.objects['values'])} objects in the image.")
            for obj in result.objects['values']:
                if 'boundingBox' in obj:
                    bounding_box = obj['boundingBox']
                    tags = obj.get('tags', [])
                    confidence = tags[0]['confidence'] if tags else None
                    tag_name = tags[0]['name'] if tags else 'unknown'

                    objs.append({
                        "box": [
                            bounding_box['x'], bounding_box['y'],
                            bounding_box['x'] + bounding_box['w'],
                            bounding_box['y'] + bounding_box['h']
                        ],
                        "confidence": confidence,
                        "tag": tag_name
                    })
                    logger.info(f"Object detected: {tag_name} with confidence {confidence} and bounding box {bounding_box}")
                else:
                    logger.warning(f"Object does not have boundingBox: {obj}")
        else:
            logger.warning("No objects detected in the image.")

        return objs
    except KeyError as e:
        logger.error(f"Key error accessing result data: {e}")
        return []
    except Exception as e:
        logger.error(f"Error generating tags and boxes: {e}")
        return []
    

def sam(image: str, bounding_box: List[int]) -> Optional[bytes]:
    """
    Runs the SAM (Segment Anything Model) on an image using specified bounding box coordinates.

    :param image: Base64 encoded string of the image.
    :param bounding_box: Coordinates for the bounding box [x1, y1, x2, y2].
    :return: The result from the SAM model as bytes, or None on failure.
    :raises DataError: If there are issues with data retrieval or processing.
    """
    allow_self_signed_https(True)
    data = {
        "input_data": {
            "columns": ["image", "input_points", "input_boxes", "input_labels", "multimask_output"],
            "index": [0],
            "data": [[image, "", f"[{bounding_box}]", "", False]]
        },
        "params": {}
    }

    url = os.getenv('SAM_URL')
    api_key = os.getenv('SAM_API_KEY')
    return make_request(url, data, api_key)


def save_image_locally(image_bytes: bytes, image_path: str) -> str:
    """
    Saves an image to the local filesystem using the specified file path and returns the path.

    :param image_bytes: The raw bytes of the image to save.
    :param image_path: The path to save the image to.
    :return: The path where the image was saved.
    :raises ValueError: If there is an error saving the image.
    """
    try:
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        return image_path
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        raise ValueError("Failed to save image")

def edit_image(image_path: str, prompt: str) -> str:
    """
    Edits an image based on a specified prompt by applying object detection, segmentation,
    and inpainting techniques. This function automates the process of identifying objects
    and modifying them according to the prompt, yielding a newly generated image. The edited
    image is saved locally and the function returns the path to the saved image.

    :param image_path: Path to the image file to be edited.
    :param prompt: Description of the desired changes to be applied to the image.
    :return: The local path where the edited image is saved.
    :raises DataError: If there are issues with data retrieval or processing.
    """
    try:
        logger.info(f"Starting image editing process for image: {image_path} with prompt: {prompt}")

        bounding_boxes = generate_tags_and_boxes(image_path)
        if not bounding_boxes:
            logger.info("No bounding boxes found")
            return

        bounding_box = bounding_boxes[0]["box"]
        logger.info(f"Bounding box found: {bounding_box}")

        logger.info("Running SAM...")
        sam_result = sam(image_to_base64(image_path), bounding_box)

        if not sam_result:
            logger.error("SAM failed")
            return

        sam_data = json.loads(sam_result.decode('utf-8'))
        mask_base64 = sam_data[0]["response"]["predictions"][0]["masks_per_prediction"][0]["encoded_binary_mask"]
        logger.info("SAM completed successfully")

        logger.info("Running Inpainting...")
        inpainting_result = inpainting(image_to_base64(image_path), mask_base64, prompt)

        if not inpainting_result:
            logger.error("Inpainting failed")
            return

        inpainting_data = json.loads(inpainting_result.decode('utf-8'))
        generated_image_base64 = inpainting_data[0]["generated_image"]
        logger.info("Inpainting completed successfully")

        # Decode the base64 image
        generated_image_bytes = base64.b64decode(generated_image_base64)

        # Define the path to save the edited image
        edited_image_path = "edited_image.jpg"

        # Save the image locally
        with open(edited_image_path, 'wb') as f:
            f.write(generated_image_bytes)

        logger.info(f"Edited image saved to: {edited_image_path}")
        return edited_image_path

    except Exception as e:
        logger.error(f"Error editing image: {e}")
        logger.error(traceback.format_exc())
        raise





