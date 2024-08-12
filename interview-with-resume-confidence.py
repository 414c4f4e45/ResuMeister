import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import random
import re

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Initialize GPT-2 for generating follow-up questions
gpt2_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def generate_follow_up_question(answer_text, resume_text):
    # Create a context snippet based on key details from the resume
    snippet = create_context_snippet(resume_text)
    
    prompt = f"Based on the following resume snippet and the answer given, generate a relevant follow-up question.\nResume Snippet: {snippet}\nAnswer: {answer_text}\nFollow-up Question:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7, max_new_tokens=50)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the question from the prompt
    question = question.replace(prompt, "").strip()
    return question if question else "Can you provide more details?"

def create_context_snippet(resume_text):
    # Extract key information using regex or other methods
    # For simplicity, let's extract some key phrases and skills
    keywords = re.findall(r'\b\w+\b', resume_text)
    snippet = " ".join(keywords[:200])  # Create a snippet with the first 200 words
    return snippet

def get_sentiment_score(text):
    # Tokenize the text and split if necessary
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length]  # Truncate to the maximum length
    sentiment = sentiment_pipeline(text)
    return sentiment[0]

# Streamlit app
st.title("AI Interviewer")

# Upload the resume PDF
uploaded_file = st.file_uploader("Upload your resume in PDF format", type="pdf")

# Session state initialization
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""

# Define the initial set of questions
initial_questions = [
    "What are your technical skills?",
    "What is your educational background?",
    "Can you describe your work experience?",
    "What projects have you worked on?",
    "What certifications do you hold?"
]

# Extract resume text if a file is uploaded
if uploaded_file is not None and st.session_state.resume_text is None:
    st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
    # Compute sentiment score for the resume text
    sentiment_score = get_sentiment_score(st.session_state.resume_text)
    st.write(f"Resume Sentiment Score: {sentiment_score['label']} with confidence {sentiment_score['score']:.2f}")

# Randomly select an initial question
if st.session_state.resume_text:
    if st.session_state.current_question is None and not st.session_state.conversation:
        if initial_questions:
            st.session_state.current_question = random.choice(initial_questions)
            st.session_state.conversation.append({
                "question": st.session_state.current_question,
                "answer": ""
            })

    # Display the current question
    if st.session_state.current_question:
        st.write(f"Interviewer: {st.session_state.current_question}")

        # Create a form to handle the interview questions
        with st.form(key='interview_form'):
            user_answer = st.text_input("Your Answer:", key="user_answer_input")
            submit_button = st.form_submit_button(label='Submit Answer')

            if submit_button:
                if user_answer:
                    # Save the user's answer to the conversation history
                    st.session_state.conversation[-1]["answer"] = user_answer

                    # Generate the next question based on the user's answer and resume text
                    follow_up_question = generate_follow_up_question(user_answer, st.session_state.resume_text)
                    st.session_state.current_question = follow_up_question
                    st.session_state.conversation.append({
                        "question": follow_up_question,
                        "answer": ""
                    })

                    # Clear the input box by re-running the script
                    st.rerun()

# Stop interview button
if st.button("Stop Interview"):
    st.write("Interview stopped. Thank you for participating!")
    st.stop()

# Display the conversation history
if st.session_state.conversation:
    st.write("Conversation History:")
    for i, qa_pair in enumerate(st.session_state.conversation):
        st.write(f"Q{i+1}: {qa_pair['question']}")
        if qa_pair['answer']:
            st.write(f"A{i+1}: {qa_pair['answer']}")
