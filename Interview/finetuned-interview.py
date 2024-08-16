import streamlit as st
from PyPDF2 import PdfReader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import re
import os

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Define the directory of your fine-tuned model
fine_tuned_model_dir = r"../fine-tuned-gpt2"

# Verify that the path exists
if not os.path.isdir(fine_tuned_model_dir):
    raise ValueError(f"The directory {fine_tuned_model_dir} does not exist.")

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_dir)
gpt2_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_dir)

def generate_follow_up_question(answer_text, resume_text, asked_questions):
    # Create a context snippet based on key details from the resume
    snippet = create_context_snippet(resume_text)
    
    prompt = f"Based on the following resume snippet and the answer given, generate a relevant and professional follow-up question. The question should be closely related to the user's response and should avoid being generic or irrelevant.\nResume Snippet: {snippet}\nAnswer: {answer_text}\nPrevious Questions: {'; '.join(asked_questions)}\nFollow-up Question:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate follow-up question
    outputs = gpt2_model.generate(
        inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the question from the prompt
    question_start = generated_text.find("Follow-up Question:") + len("Follow-up Question:")
    question = generated_text[question_start:].strip()

    # Use regex to capture only the first sentence after "Follow-up Question:"
    question = re.split(r'[?.!]', question)[0].strip() + '?'

    # Check if the question is too generic or irrelevant
    if "resume" in question.lower() or "job" in question.lower() or len(question.split()) < 4:
        question = generate_fallback_question(asked_questions)

    # If the generated question is similar to a previous one, return a fallback question
    if question in asked_questions:
        question = generate_fallback_question(asked_questions)

    return question if question else "Can you provide more details?"

def create_context_snippet(resume_text):
    # Extract key information using regex or other methods
    # For simplicity, let's extract some key phrases and skills
    keywords = re.findall(r'\b\w+\b', resume_text)
    snippet = " ".join(keywords[:200])  # Create a snippet with the first 200 words
    return snippet

def generate_fallback_question(asked_questions):
    fallback_questions = [
        "Can you tell me more about your experience with AI?",
        "What projects are you most proud of?",
        "How do you stay updated with the latest trends in your field?",
        "Can you describe a challenging problem you've solved?",
        "What are your career goals?"
    ]
    # Select a fallback question that hasn't been asked yet
    remaining_questions = [q for q in fallback_questions if q not in asked_questions]
    if remaining_questions:
        return random.choice(remaining_questions)
    else:
        return "What else can you share about your background?"

# Function to get the answer to a question
def ask_question(resume_text, question):
    result = qa_pipeline(question=question, context=resume_text)
    return result['answer']

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
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
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

# Randomly select an initial question
if st.session_state.resume_text:
    if st.session_state.current_question is None and not st.session_state.conversation:
        if initial_questions:
            st.session_state.current_question = random.choice(initial_questions)
            st.session_state.conversation.append({
                "question": st.session_state.current_question,
                "answer": ""
            })
            st.session_state.asked_questions.append(st.session_state.current_question)

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
                    follow_up_question = generate_follow_up_question(user_answer, st.session_state.resume_text, st.session_state.asked_questions)
                    st.session_state.current_question = follow_up_question
                    st.session_state.conversation.append({
                        "question": follow_up_question,
                        "answer": ""
                    })
                    st.session_state.asked_questions.append(follow_up_question)

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
