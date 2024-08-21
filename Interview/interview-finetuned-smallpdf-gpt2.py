import streamlit as st
import tempfile
import shutil
import string
import random
import re
import os

from smallpdf import SmallPDF
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

# Initialize the GPT-2 model, tokenizer, and sentiment analysis
fine_tuned_model_dir = r"../fine-tuned-gpt2"
if not os.path.isdir(fine_tuned_model_dir):
    raise ValueError(f"The directory {fine_tuned_model_dir} does not exist.")
    
tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_dir)
gpt2_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_dir)

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
# Function to calculate sentiment and confidence score
def get_sentiment_score(text):
    max_length = 512  # Define maximum length for the sentiment model
    text = text[:max_length]  # Truncate text to fit model's input size
    result = sentiment_pipeline(text)
    sentiment = result[0]['label'].capitalize()  # Capitalize only the first letter
    confidence = result[0]['score'] * 100  # Convert to percentage
    return sentiment, confidence


# Streamlit app
st.title("PDF Uploader")
job_role = st.text_area("Enter the Job Title/Role", height=40) 
job = st.text_area("Enter the Job Description", height=200)
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None and "uploaded_file_name" in st.session_state:
    if uploaded_file.name != st.session_state.uploaded_file_name:
        st.session_state.extract = None
        st.session_state.extracted = False
        st.session_state.error_occurred = False
        st.session_state.temp_file_path = None
        st.session_state.show_skills = False
        st.session_state.show_experience = False

if uploaded_file is not None:
    st.session_state.uploaded_file_name = uploaded_file.name

# Initialize session state variables
if "extract" not in st.session_state:
    st.session_state.extract = None
if "extracted" not in st.session_state:
    st.session_state.extracted = False
if "error_occurred" not in st.session_state:
    st.session_state.error_occurred = False
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "show_skills" not in st.session_state:
    st.session_state.show_skills = False
if "show_experience" not in st.session_state:
    st.session_state.show_experience = False
if "input_history" not in st.session_state:
    st.session_state.input_history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": "What role are you aiming for, and how does it align with your career goals?"})
if "sentiment_score" not in st.session_state:
    st.session_state.sentiment_score = None
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []
if "asked_topics" not in st.session_state:
    st.session_state.asked_topics = []
if "resume_extract" not in st.session_state:
	st.session_state.resume_extract = ""


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def run_extraction():
    try:
        # Initialize and run the smallpdf extraction
        st.session_state.extract = SmallPDF(st.session_state.temp_file_path,f"{job_role}: {job}")
        st.session_state.extract.run()
        st.session_state.extracted = True
        st.session_state.error_occurred = False
		# Formatting skills
        skills_data = "Skills:\n"
        for key, values in st.session_state.extract.skills.items():
            skills_data += f"{key}\n"
            skills_data += ','.join(values) + "\n"

        experience_data = "Experience:\n"
        # Assuming 'experience' is a dictionary similar to 'skills'
        for key, values in st.session_state.extract.experience.items():
            experience_data += f"{key}\n"
            for value in values:
                experience_data += f"{value}\n"

        # Combining both strings
        st.session_state.resume_extract = skills_data + experience_data

    except Exception as e:
        st.session_state.extracted = False
        st.session_state.error_occurred = True
        st.error(f"An error occurred: {e}")

def generate_follow_up_question(answer_text, resume_text, asked_questions, asked_topics, job_description, skills):
    #print("-----\n", combined_str, "\n------")
    prompt = (f"Based on the following resume snippet and the answer given, generate a relevant and professional follow-up question. "
              f"The question should be closely related to the user's response and should align with the job description provided.\n"
              f"Resume Snippet: {resume_text}\nAnswer: {answer_text}\nPrevious Questions: {'; '.join(asked_questions)}\n"
              f"Asked Topics: {'; '.join(asked_topics)}\nJob Description: {job_description}\nSkills: {', '.join(skills)}\nFollow-up Question:")
    
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
    
    topics = [
        'AI', 'API', 'Blockchain', 'C', 'Cloud', 'C++', 'Cryptography', 'CSS', 'Computer Vision',
        'Cyber Security', 'Data Analysis', 'Data Engineering', 'Data Science', 'Data Visualization',
        'DBMS', 'DevOps', 'Deep Learning', 'DSA', 'Encoder Decoder', 'Express JS', 'Flutter', 'Git',
        'Go', 'HTML', 'iOS', 'Java', 'JavaScript', 'Kotlin', 'Linux', 'LLM', 'ML', 'MLOps', 'MongoDB',
        'Networking', 'Neural Network', 'NLP', 'Node JS', 'OOP', 'OS', 'Python', 'R', 'RAG', 'React JS',
        'Redis', 'Rust', 'SQL', 'Statistics', 'Swift', 'System Design', 'Transformer', 'UI', 'Unity',
        'Unreal', 'UX'
    ]
    
    remaining_questions = [q for q in fallback_questions if q not in asked_questions]
    if remaining_questions:
        return random.choice(remaining_questions)
    else:
        new_topic = random.choice([t for t in topics if t not in asked_topics])
        return f"Can you describe your experience with {new_topic}?"

if uploaded_file is not None:
	if st.session_state.temp_file_path is None:
		with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
			shutil.copyfileobj(uploaded_file, temp_file)
			st.session_state.temp_file_path = temp_file.name

	st.write("Thank you for uploading the PDF!")
	st.write(f"File saved at: {st.session_state.temp_file_path}")

	if st.session_state.extract is None or st.session_state.error_occurred:
		run_extraction()
	if st.session_state.extracted:
		st.write("Extraction successful!")
		st.write(f"Your Resume Score: {st.session_state.extract.score['Score'][0]}")
		job_desc = st.session_state.extract.job_desc
		job_req = "Requirements: "+';'.join(k+':'+', '.join(v) for k,v in job_desc.items())

		def chat_actions():
			user_input = st.session_state["chat_input"]

			sentiment, confidence = get_sentiment_score(user_input)
			st.session_state.sentiment_score = {"sentiment": sentiment, "confidence": confidence}

			st.session_state["chat_history"].append({"role": "user", "content": user_input, "sentiment": sentiment})

			follow_up_question = generate_follow_up_question(
				answer_text=user_input,
				resume_text=st.session_state.resume_extract,
				asked_questions=[entry["content"] for entry in st.session_state["chat_history"] if entry["role"] == "assistant"],
				asked_topics=[],
				job_description=job_role + ': ' + job_req,
				skills=[]
			)
			st.session_state.asked_questions.append(follow_up_question)
			follow_up_question_lower = follow_up_question.lower()
			# Check if any value in the dictionary is in the follow-up question

			if any(value.lower() in follow_up_question_lower for value in st.session_state.resume_extract):
				topic = "Technical Skills"
			else:
				topic = "General"
			st.session_state.asked_topics.append(topic)
			st.session_state["chat_history"].append({"role": "assistant", "content": follow_up_question})

		st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

		for i in st.session_state["chat_history"]:
			with st.chat_message(name=i["role"]):
				st.write(i["content"])

	else:
		st.error("Extraction failed. Please try again.")
		if st.button("Retry"):
			run_extraction()

