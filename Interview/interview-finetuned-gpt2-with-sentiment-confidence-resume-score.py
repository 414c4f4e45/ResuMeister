import streamlit as st
from PyPDF2 import PdfReader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
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

# Function to extract skills or topics from job description
def extract_skills_from_job_description(job_description):
    # For simplicity, we'll just split the job description into keywords
    keywords = re.findall(r'\b\w+\b', job_description)
    return set(keywords)

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

# Initialize sentence transformer model for resume-job description matching
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to calculate sentiment and confidence score
def get_sentiment_score(text):
    max_length = 512  # Define maximum length for the sentiment model
    text = text[:max_length]  # Truncate text to fit model's input size
    result = sentiment_pipeline(text)
    sentiment = result[0]['label'].capitalize()  # Capitalize only the first letter
    confidence = result[0]['score'] * 100  # Convert to percentage
    return sentiment, confidence

# Function to calculate match score between resume and job description
def calculate_match_score(resume_text, job_description):
    resume_embedding = sentence_model.encode(resume_text, convert_to_tensor=True)
    job_description_embedding = sentence_model.encode(job_description, convert_to_tensor=True)
    match_score = util.pytorch_cos_sim(resume_embedding, job_description_embedding).item()
    return match_score * 100  # Convert to percentage

def generate_follow_up_question(answer_text, resume_text, asked_questions, asked_topics, job_description, skills):
    snippet = create_context_snippet(resume_text)
    
    prompt = f"Based on the following resume snippet and the answer given, generate a relevant and professional follow-up question. The question should be closely related to the user's response and should align with the job description provided.\nResume Snippet: {snippet}\nAnswer: {answer_text}\nPrevious Questions: {'; '.join(asked_questions)}\nAsked Topics: {'; '.join(asked_topics)}\nJob Description: {job_description}\nSkills: {', '.join(skills)}\nFollow-up Question:"
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
              'Data Engineering',
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

job_description = st.text_area("Enter the Job Description", height=200)
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
if "match_score" not in st.session_state:
    st.session_state.match_score = None
if "skills" not in st.session_state:
    st.session_state.skills = None

# Handle job description extraction and skills
if job_description:
    st.session_state.skills = extract_skills_from_job_description(job_description)

# Handle resume text extraction
if uploaded_file is not None and st.session_state.resume_text is None:
    st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
    resume_sentiment, resume_confidence = get_sentiment_score(st.session_state.resume_text)
    st.session_state.resume_sentiment = resume_sentiment
    st.session_state.resume_confidence = resume_confidence

    if job_description:
        st.session_state.match_score = calculate_match_score(st.session_state.resume_text, job_description)

# Display resume sentiment and match score if resume is uploaded
if st.session_state.resume_text:
    st.write(f"Resume Sentiment: {st.session_state.resume_sentiment}")
    st.write(f"Confidence: {st.session_state.resume_confidence:.2f}%")
    if st.session_state.match_score is not None:
        st.write(f"Match Score with Job Description: {st.session_state.match_score:.2f}%")

# Handle question generation and display
if st.session_state.current_question is None and not st.session_state.conversation:
    if job_description and st.session_state.resume_text:
        initial_questions = [
            "What are your technical skills?",
            "What is your educational background?",
            "Can you describe your work experience?",
            "What projects have you worked on?",
            "What certifications do you hold?"
        ]
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
    st.write(f"Current Question: {st.session_state.current_question}")
    user_answer = st.text_area("Your Answer", key="answer_text")

    if st.button("Submit Answer"):
        sentiment, confidence = get_sentiment_score(user_answer)
        follow_up_question = generate_follow_up_question(
            answer_text=user_answer,
            resume_text=st.session_state.resume_text,
            asked_questions=st.session_state.asked_questions,
            asked_topics=st.session_state.asked_topics,
            job_description=job_description,
            skills=st.session_state.skills
        )
        
        st.session_state.conversation[-1]["answer"] = user_answer
        st.session_state.conversation[-1]["sentiment"] = sentiment
        st.session_state.conversation[-1]["confidence"] = confidence

        st.session_state.current_question = follow_up_question
        st.session_state.conversation.append({
            "question": follow_up_question,
            "answer": ""
        })
        st.session_state.asked_questions.append(follow_up_question)
        topic = "Technical Skills" if "skills" in follow_up_question.lower() else "General"
        st.session_state.asked_topics.append(topic)

# Display conversation history
if st.session_state.conversation:
    st.write("Interview Conversation History")
    for i, entry in enumerate(st.session_state.conversation):
        question = entry.get("question", "N/A")
        answer = entry.get("answer", "N/A")
        sentiment = entry.get("sentiment", "N/A").capitalize()
        confidence = entry.get("confidence", "N/A")

        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = "N/A"
        
        if isinstance(confidence, float):
            confidence_display = f"{confidence:.2f}%"
        else:
            confidence_display = confidence

        st.write(f"Q{i+1}: {question}")
        st.write(f"A{i+1}: {answer} (Sentiment: {sentiment}, Confidence: {confidence_display})")
