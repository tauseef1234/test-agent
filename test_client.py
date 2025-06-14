import requests
import sys

# This interactive script simulates a user conversation, correctly handling the session ID.
BASE_URL = "http://localhost:8000"
ENDPOINT = "/bank-rag-agent"
URL = f"{BASE_URL}{ENDPOINT}"

def post_message(text: str, session_id: str | None = None) -> tuple[str, str]:
    """Sends a message to the chatbot API and returns the bot's response and session ID."""
    payload = {"text": text}
    if session_id:
        payload["session_id"] = session_id
    
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # Print the bot's response
        print(f"\n< Bot: {data['output']}")
        
        # IMPORTANT: Return the session_id received from the server
        return data['output'], data['session_id']
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to the API: {e}")
        print("Please ensure the FastAPI server is running.")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting interactive chatbot client.")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    # Initialize session_id to None to start a new conversation
    current_session_id = None
    
    # Start the conversation with a generic greeting
    print("< Bot: Sending initial greeting...")
    bot_output, current_session_id = post_message("Hello", session_id=current_session_id)
    
    # Loop indefinitely to chat with the bot
    while True:
        try:
            # Get user input from the terminal
            user_input = input("\n> You: ")
            
            if user_input.lower() == 'quit':
                print("< Client: Exiting chat.")
                break
                
            # Post the message and update the session_id for the next loop
            bot_output, current_session_id = post_message(user_input, session_id=current_session_id)

        except KeyboardInterrupt:
            print("\n< Client: Exiting chat.")
            break