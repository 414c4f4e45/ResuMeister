import os
from openai import OpenAI

# Initialize the OpenAI client with the API key from the environment variable
client = OpenAI(
    api_key=os.environ.get("Your OpenAI API Key")  # Make sure the environment variable is set
    # Get your API key from "https://platform.openai.com/api-keys"
)

# Initial system message
messages = [
    {"role": "system", "content": "You're a recruiter who asks interview questions. You ask one new question after my response."}
]

while True:
    # Get user input
    content = input("User: ")
    messages.append({"role": "user", "content": content})

    # Create a chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    # Extract the assistant's response
    chat_response = chat_completion['choices'][0]['message']['content']
    print(f'ChatGPT: {chat_response}')
    messages.append({"role": "assistant", "content": chat_response})
