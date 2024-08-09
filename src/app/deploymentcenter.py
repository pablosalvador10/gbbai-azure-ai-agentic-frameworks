import os
from typing import Optional

# Load environment variables
import streamlit as st
from src.app.managers import create_azure_openai_manager


def load_default_deployment(
    name: Optional[str] = None,
    key: Optional[str] = None,
    endpoint: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    """
    Load default deployment settings, optionally from provided parameters.

    Ensures that a deployment with the same name does not already exist.

    :param name: (Optional) Name of the deployment.
    :param key: (Optional) Azure OpenAI key.
    :param endpoint: (Optional) API endpoint for Azure OpenAI.
    :param version: (Optional) API version for Azure OpenAI.
    :raises ValueError: If required deployment settings are missing.
    """
    # Ensure deployments is a dictionary
    if "deployments" not in st.session_state or not isinstance(
        st.session_state.deployments, dict
    ):
        st.session_state.deployments = {}

    # Check if the deployment name already exists
    deployment_name = (
        name if name else os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
    )
    if deployment_name in st.session_state.deployments:
        return  # Exit the function if deployment already exists

    default_deployment = {
        "name": deployment_name,
        "key": key if key else os.getenv("AZURE_OPENAI_KEY"),
        "endpoint": endpoint if endpoint else os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "version": version if version else os.getenv("AZURE_OPENAI_API_VERSION"),
        "stream": False,
    }

    if all(
        value is not None for value in default_deployment.values() if value != False
    ):
        st.session_state.deployments[default_deployment["name"]] = default_deployment



def add_deployment_aoai_form() -> None:
    """
    Render the form to add a new Azure OpenAI deployment.

    This function provides a form in the Streamlit sidebar to add a new deployment, allowing users to specify deployment details.
    """
    with st.form("add_deployment_aoai_form"):
        deployment_name = st.text_input(
            "Deployment id",
            help="Enter the deployment ID for Azure OpenAI.",
            placeholder="e.g., chat-gpt-1234abcd",
        )
        deployment_key = st.text_input(
            "Azure OpenAI Key",
            help="Enter your Azure OpenAI key.",
            type="password",
            placeholder="e.g., sk-ab*****..",
        )
        deployment_endpoint = st.text_input(
            "API Endpoint",
            help="Enter the API endpoint for Azure OpenAI.",
            placeholder="e.g., https://api.openai.com/v1",
        )
        deployment_version = st.text_input(
            "API Version",
            help="Enter the API version for Azure OpenAI.",
            placeholder="e.g., 2024-02-15-preview",
        )
        is_streaming = st.radio(
            "Streaming",
            (True, False),
            index=1,
            format_func=lambda x: "Yes" if x else "No",
            help="Select 'Yes' if the model will be tested with output in streaming mode.",
        )
        submitted = st.form_submit_button("Add Deployment")

        if submitted:
            if (
                deployment_name
                and deployment_key
                and deployment_endpoint
                and deployment_version
            ):
                if "deployments" not in st.session_state:
                    st.session_state.deployments = {}

                try:
                    test_client = create_azure_openai_manager(
                        api_key=deployment_key,
                        azure_endpoint=deployment_endpoint,
                        api_version=deployment_version,
                        deployment_id=deployment_name,
                    )

                    stream = test_client.openai_client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {"role": "system", "content": "Test: Verify setup."},
                            {"role": "user", "content": "test"},
                        ],
                        max_tokens=2,
                        seed=555,
                        stream=is_streaming,
                    )
                except Exception as e:
                    st.warning(
                        f"""An issue occurred while initializing the Azure OpenAI manager. {e} Please try again. If the issue persists,
                                    verify your configuration."""
                    )
                    return

                if deployment_name not in st.session_state.deployments:
                    st.session_state.deployments[deployment_name] = {
                        "key": deployment_key,
                        "endpoint": deployment_endpoint,
                        "version": deployment_version,
                        "stream": is_streaming,
                    }
                    st.toast(f"Deployment '{deployment_name}' added successfully.")
                    st.rerun()
                else:
                    st.error(
                        f"A deployment with the name '{deployment_name}' already exists."
                    )


def display_deployments() -> None:
    """
    Display and manage existing Azure OpenAI deployments.

    This function renders the existing deployments in the Streamlit sidebar, allowing users to view, update, or remove deployments.
    """
    if "deployments" in st.session_state:
        st.markdown("##### Loaded Deployments")
        if st.session_state.deployments == {}:
            st.sidebar.error(
                "No deployments were found. Please add a deployment in the Deployment Center."
            )
        else:
            for deployment_name, deployment in st.session_state.deployments.items():
                with st.expander(deployment_name):
                    updated_name = st.text_input(
                        "Name", value=deployment_name, key=f"name_{deployment_name}"
                    )
                    updated_key = st.text_input(
                        "Key",
                        value=deployment.get("key", ""),
                        type="password",
                        key=f"key_{deployment_name}",
                    )
                    updated_endpoint = st.text_input(
                        "Endpoint",
                        value=deployment.get("endpoint", ""),
                        key=f"endpoint_{deployment_name}",
                    )
                    updated_version = st.text_input(
                        "Version",
                        value=deployment.get("version", ""),
                        key=f"version_{deployment_name}",
                    )
                    updated_stream = st.radio(
                        "Streaming",
                        (True, False),
                        format_func=lambda x: "Yes" if x else "No",
                        index=0 if deployment.get("stream", False) else 1,
                        key=f"stream_{deployment_name}",
                        help="Select 'Yes' if the model will be tested with output in streaming mode.",
                    )

                    if st.button("Update Deployment", key=f"update_{deployment_name}"):
                        st.session_state.deployments[deployment_name] = {
                            "key": updated_key,
                            "endpoint": updated_endpoint,
                            "version": updated_version,
                            "stream": updated_stream,
                        }
                        st.rerun()

                    if st.button("Remove Deployment", key=f"remove_{deployment_name}"):
                        del st.session_state.deployments[deployment_name]
                        st.rerun()
    else:
        st.sidebar.error(
            "No deployments were found. Please add a deployment in the Deployment Center."
        )


def create_benchmark_center() -> None:
    """
    Creates a benchmark center UI component in a Streamlit application.
    This component allows users to add their MaaS Deployment for benchmarking
    against different model families.

    The function dynamically generates UI elements based on the user's selection
    of model family from a dropdown menu. Currently, it supports the "AOAI" model
    family and provides a placeholder for future expansion to other model families.
    """
    with st.expander("➕ Add Your MaaS Deployment", expanded=False):
        operation = st.selectbox(
            "Choose Model Family:",
            ("AOAI", "Other"),
            index=0,
            help="Select the benchmark you want to perform to evaluate AI model performance.",
            placeholder="Select a Benchmark",
        )
        if operation == "AOAI":
            add_deployment_aoai_form()  # This function needs to be defined elsewhere in your code.
        else:
            st.info("Other deployment options will be available soon.")