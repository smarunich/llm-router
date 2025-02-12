from openai import OpenAI
import os

PROMPT = "Can you write me a song?  Use as many emojis as possible."

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY", ""),
    max_retries=0,
    base_url="http://0.0.0.0:8084/v1/"
)

messages=[
    {
        "role":"user",
        "content": PROMPT
    }
]

extra_body = {
    "nim-llm-router": {
        "policy": "task_router",
        "routing_strategy": "triton"
    }
}

def handle_non_streaming():
    """Handle non-streaming completion with usage reporting"""
    print("\n=== Non-Streaming Mode ===")
    
    chat_completion = client.chat.completions.with_raw_response.create(
        messages=messages,
        model="",
        extra_body=extra_body
    )

    # Print classifier info
    chosen_classifier = chat_completion.headers.get('X-Chosen-Classifier')
    print(f"Chosen Classifier: {chosen_classifier}")

    # Parse the response
    completion = chat_completion.parse()
    
    # Print the completion
    print("\nCompletion:")
    print(completion.choices[0].message.content)

    # Print usage information
    usage = completion.usage
    print("\nToken Usage:")
    print(f"  Prompt tokens: {usage.prompt_tokens}")
    print(f"  Completion tokens: {usage.completion_tokens}")
    print(f"  Total tokens: {usage.total_tokens}")

def handle_streaming():
    """Handle streaming completion with usage reporting"""
    print("\n=== Streaming Mode ===")
    
    chat_completion = client.chat.completions.with_raw_response.create(
        messages=messages,
        model="",
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body
    )

    # Print classifier info
    chosen_classifier = chat_completion.headers.get('X-Chosen-Classifier')
    print(f"Chosen Classifier: {chosen_classifier}")

    # Get the response iterator
    stream = chat_completion.parse()
    
    print("\nCompletion:")
    # Process the stream
    try:
        for chunk in stream:
            # Update final usage if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                final_usage = chunk.usage
            
            # Only try to print content if delta exists and has content
            if (len(chunk.choices) > 0 and  # Check if choices exists and has elements
                hasattr(chunk.choices[0], 'delta') and 
                hasattr(chunk.choices[0].delta, 'content') and 
                chunk.choices[0].delta.content is not None):
                print(chunk.choices[0].delta.content, end="", flush=True)
    except Exception as e:
        pass
    
    # Print usage information once at the end
    if final_usage:
        print("\n\nToken Usage:")
        print(f"  Prompt tokens: {final_usage.prompt_tokens}")
        print(f"  Completion tokens: {final_usage.completion_tokens}")
        print(f"  Total tokens: {final_usage.total_tokens}")

def main():
    """Run both streaming and non-streaming examples"""
    try:
        # Test non-streaming mode
        handle_non_streaming()
        
        # Test streaming mode
        handle_streaming()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
