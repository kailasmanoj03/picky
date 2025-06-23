import streamlit as st
import openai
import time
import json

# --- Helper Functions ---

# Define the function that the agent can call
def send_email(to: str, subject: str, body: str):
    """
    Sends an email to a specified recipient.

    Args:
        to (str): The email address of the recipient.
        subject (str): The subject of the email.
        body (str): The body content of the email.
    """
    # In a real app, you'd use a service like SendGrid or SMTP.
    # For this demo, we'll just print to the console and return a success message.
    print(f"--- MOCK EMAIL ---")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    print(f"------------------")
    return "Email sent successfully!"

# --- Streamlit App ---

st.set_page_config(page_title="Context-Aware Assistant", layout="wide")
st.title("📄 Context-Aware Assistant with Function Calling")
st.markdown("This assistant is trained on your context and can send emails on your behalf.")

# --- Initialization & Sidebar ---

# Initialize session state variables
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "email_list" not in st.session_state:
    st.session_state.email_list = []

with st.sidebar:
    st.header("Configuration")

    # 1. API Key Input
    try:
        # Securely get the API key from Streamlit's secrets
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        st.success("OpenAI API key loaded from secrets.", icon="✅")
    except (KeyError, FileNotFoundError):
        st.error("`OPENAI_API_KEY` not found in secrets.toml. Please add it.")
        st.stop()


    # 2. Training Context Input
    st.subheader("1. Training Context")
    context = st.text_area(
        "Provide context for the assistant (e.g., FAQs, product info).",
        height=250,
        key="context_input"
    )

    if st.button("Create or Update Assistant"):
        if not context.strip():
            st.warning("Please provide some context before creating the assistant.")
        else:
            with st.spinner("Creating assistant and uploading file..."):
                # Upload the context as a file to OpenAI
                context_file = client.files.create(
                    file=("context.txt", context.encode('utf-8')),
                    purpose='assistants'
                )

                # Define the tools for the assistant
                tools = [
                    {"type": "retrieval"}, # For answering questions from the file
                    {
                        "type": "function",
                        "function": {
                            "name": "send_email",
                            "description": "Sends an email to a specified recipient. Use this for any user request that involves sending information to an email address.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "to": {"type": "string", "description": "The email address of the recipient."},
                                    "subject": {"type": "string", "description": "The subject line of the email."},
                                    "body": {"type": "string", "description": "The main content/body of the email."},
                                },
                                "required": ["to", "subject", "body"]
                            }
                        }
                    }
                ]

                # Create the assistant
                assistant = client.beta.assistants.create(
                    name="Context-Restricted Assistant",
                    instructions="You are a helpful assistant. Answer questions ONLY based on the uploaded training content file. If the user asks something you cannot answer from the context, respond with 'I'm sorry, I can only answer questions based on the provided training content.' If the user asks you to send an email, use the `send_email` function. You must have the recipient's email address to use this function.",
                    tools=tools,
                    model="gpt-4-1106-preview",
                    file_ids=[context_file.id]
                )
                st.session_state.assistant_id = assistant.id

                # Create a thread for the conversation
                thread = client.beta.threads.create()
                st.session_state.thread_id = thread.id
                st.session_state.messages = [] # Reset messages
                st.success(f"Assistant created with ID: `{assistant.id}`", icon="🤖")
                st.info(f"A new conversation thread has been started: `{thread.id}`", icon="🧵")


    # 3. Email Capture
    st.subheader("2. Email Management")
    email_input = st.text_input("Enter an email to add to the list:")
    if st.button("Add Email"):
        if email_input:
            st.session_state.email_list.append(email_input)
            st.success(f"Added `{email_input}` to the list.")
        else:
            st.warning("Please enter an email address.")

    if st.session_state.email_list:
        st.write("Available Emails:")
        st.json(st.session_state.email_list)


# --- Main Chat Interface ---

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle chat input
if prompt := st.chat_input("Ask a question or request to send an email..."):
    if not st.session_state.assistant_id or not st.session_state.thread_id:
        st.error("Please create an assistant first using the sidebar.")
        st.stop()

    # Add user's message to the state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add the message to the OpenAI thread
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=prompt
    )

    # Create and run the assistant
    run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id,
    )

    # Wait for the run to complete, handling function calls
    with st.spinner("Assistant is thinking..."):
        while run.status in ['queued', 'in_progress']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )

        if run.status == "requires_action":
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "send_email":
                    st.info("Assistant is requesting to send an email...", icon="📧")
                    args = json.loads(tool_call.function.arguments)
                    output = send_email(to=args['to'], subject=args['subject'], body=args['body'])
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output,
                    })

            # Submit the function call output back to the assistant
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=st.session_state.thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            # Re-run the waiting loop for the final response
            while run.status in ['queued', 'in_progress']:
                 time.sleep(1)
                 run = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id, run_id=run.id)


    # Retrieve and display the assistant's final response
    messages = client.beta.threads.messages.list(
        thread_id=st.session_state.thread_id
    )
    assistant_message = messages.data[0].content[0].text.value

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)
