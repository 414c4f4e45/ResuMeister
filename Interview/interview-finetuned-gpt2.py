import streamlit as st
from PyPDF2 import PdfReader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
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

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Function to calculate sentiment and confidence score
def get_sentiment_score(text):
    max_length = 512  # Define maximum length for the sentiment model
    text = text[:max_length]  # Truncate text to fit model's input size
    result = sentiment_pipeline(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return sentiment, confidence

def generate_follow_up_question(answer_text, resume_text, asked_questions, asked_topics):
    snippet = create_context_snippet(resume_text)
    
    prompt = f"Based on the following resume snippet and the answer given, generate a relevant and professional follow-up question. The question should be closely related to the user's response and should avoid being generic or irrelevant. Ensure to switch topics if the user has expressed frustration or requested a topic change.\nResume Snippet: {snippet}\nAnswer: {answer_text}\nPrevious Questions: {'; '.join(asked_questions)}\nAsked Topics: {'; '.join(asked_topics)}\nFollow-up Question:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
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

    question_start = generated_text.find("Follow-up Question:") + len("Follow-up Question:")
    question = generated_text[question_start:].strip()

    question = re.split(r'[?.!]', question)[0].strip() + '?'

    if "resume" in question.lower() or "job" in question.lower() or len(question.split()) < 4:
        question = generate_fallback_question(asked_questions, asked_topics)

    if question in asked_questions:
        question = generate_fallback_question(asked_questions, asked_topics)

    return question if question else "Can you provide more details?"

def create_context_snippet(resume_text):
    keywords = re.findall(r'\b\w+\b', resume_text)
    snippet = " ".join(keywords[:200])
    return snippet

def generate_fallback_question(asked_questions, asked_topics):
    fallback_questions = [
        "Can you tell me more about your experience?",
        "What projects are you most proud of?",
        "How do you stay updated with the latest trends in your field?",
        "Can you describe a challenging problem you've solved?",
        "What are your career goals?",
        "What programming languages are you familiar with?",
        "Can you explain a recent project you worked on?",
        "How do you approach problem-solving in your projects?"
    ]
    
    topics = ['AI',
              'API',
              'Blockchain',
              'C',
              'Cloud',
              'C++',
              'Cryptography',
              'CSS',
              'Computer Vision',
              'Cyber Security',
              'Data Analysis',
              'Dada Engineering',
              'Data Science',
              'Data Visualization',
              'DBMS',
              'DevOps',
              'Deep Learning',
              'DSA',
              'Encoder Decoder',
              'Express JS',
              'Flutter',
              'Git',
              'Go',
              'HTML',
              'iOS',
              'Java',
              'JavaScript',
              'Kotlin',
              'Linux',
              'LLM',
              'ML',
              'MLOps',
              'MongoDB',
              'Networking',
              'Neural Network',
              'NLP',
              'Node JS',
              'OOP',
              'OS',
              'Python',
              'R',
              'RAG',
              'React JS',
              'Redis',
              'Rust',
              'SQL',
              'Statistics',
              'Swift',
              'System Design',
              'Transformer',
              'UI',
              'Unity',
              'Unreal',
              'UX']
    
    remaining_questions = [q for q in fallback_questions if q not in asked_questions]
    if remaining_questions:
        return random.choice(remaining_questions)
    else:
        new_topic = random.choice([t for t in topics if t not in asked_topics])
        return f"Can you describe your experience with {new_topic}?"

# Streamlit app
st.title("AI Interviewer")

uploaded_file = st.file_uploader("Upload your resume in PDF format", type="pdf")

if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "asked_topics" not in st.session_state:
    st.session_state.asked_topics = []
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""
if "resume_sentiment" not in st.session_state:
    st.session_state.resume_sentiment = None
if "resume_confidence" not in st.session_state:
    st.session_state.resume_confidence = None

initial_questions = [
    "What are your technical skills?",
    "What is your educational background?",
    "Can you describe your work experience?",
    "What projects have you worked on?",
    "What certifications do you hold?"
]

if uploaded_file is not None and st.session_state.resume_text is None:
    st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
    resume_sentiment, resume_confidence = get_sentiment_score(st.session_state.resume_text)
    st.session_state.resume_sentiment = resume_sentiment
    st.session_state.resume_confidence = resume_confidence

if st.session_state.resume_text:
    st.write(f"Resume Sentiment: {st.session_state.resume_sentiment} (Confidence: {st.session_state.resume_confidence:.2f})")

if st.session_state.resume_text:
    if st.session_state.current_question is None and not st.session_state.conversation:
        if initial_questions:
            st.session_state.current_question = random.choice(initial_questions)
            st.session_state.conversation.append({
                "question": st.session_state.current_question,
                "answer": ""
            })
            st.session_state.asked_questions.append(st.session_state.current_question)
            topic = "Technical Skills" if "skills" in st.session_state.current_question.lower() else "General"
            st.session_state.asked_topics.append(topic)

    if st.session_state.current_question:
        st.write(f"Interviewer: {st.session_state.current_question}")

        with st.form(key='interview_form'):
            user_answer = st.text_input("Your Answer:", key="user_answer_input")
            submit_button = st.form_submit_button(label='Submit Answer')

            if submit_button:
                if user_answer:
                    st.session_state.conversation[-1]["answer"] = user_answer

                    answer_sentiment, answer_confidence = get_sentiment_score(user_answer)
                    st.session_state.conversation[-1]["sentiment"] = answer_sentiment
                    st.session_state.conversation[-1]["confidence"] = answer_confidence

                    follow_up_question = generate_follow_up_question(user_answer, st.session_state.resume_text, st.session_state.asked_questions, st.session_state.asked_topics)
                    st.session_state.current_question = follow_up_question
                    st.session_state.conversation.append({
                        "question": follow_up_question,
                        "answer": ""
                    })
                    st.session_state.asked_questions.append(follow_up_question)

                    new_topic = "Technical Skills" if "skills" in follow_up_question.lower() else "General"
                    if new_topic not in st.session_state.asked_topics:
                        st.session_state.asked_topics.append(new_topic)

                    st.rerun()

if st.button("Stop Interview"):
    st.write("Interview stopped. Thank you for participating!")
    st.stop()

if st.session_state.conversation:
    st.write("Conversation History:")
    for i, qa_pair in enumerate(st.session_state.conversation):
        st.write(f"Q{i+1}: {qa_pair['question']}")
        if qa_pair['answer']:
            st.write(f"A{i+1}: {qa_pair['answer']}")
            st.write(f"Sentiment: {qa_pair.get('sentiment', 'N/A')} (Confidence: {qa_pair.get('confidence', 'N/A'):.2f})")
