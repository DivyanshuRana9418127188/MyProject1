# import requests

# def generate_response(context):
#     """
#     Generate a response from the Ollama model based on the retrieved context.
#     """
#     url = "http://localhost:11434/v1/llama3.2:1b"  # Use your Ollama model ID
#     payload = {
#         "context": context
#     }
    
#     response = requests.post(url, json=payload, )
#     return response.json()


import requests
import json

def generate_streaming_response(model_name: str, context: str):
    """
    Generate a streaming response from the local Ollama model based on the retrieved context.
    
    Args:
        model_name (str): The name of the Ollama model to use (e.g., "llama3", "mistral").
        context (str): The combined context retrieved from the database.

    Yields:
        str: Chunks of the streamed response content.

    Raises:
        requests.exceptions.RequestException: If there's an error connecting to Ollama
                                             or an HTTP error from the Ollama API.
    """
    url = "http://localhost:11434/api/chat" # Local Ollama endpoint
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": context}],
        "stream": True # Enable streaming
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        yield json_data["message"]["content"]
                    elif "done" in json_data and json_data["done"]:
                        break # End of stream
                except json.JSONDecodeError:
                    # Log a warning, but don't stop the stream unless critical
                    print(f"\nWarning in my_llm_integration.py: Failed to parse line as JSON: {line}")
                    continue
    except requests.exceptions.RequestException as e:
        # Re-raise the exception so VectorLLM.py can catch and display it using st.error
        raise e