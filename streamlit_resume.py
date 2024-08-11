import streamlit as st
import tempfile
import shutil
from smallpdf import smallpdf

# Streamlit app
st.title("PDF Uploader")

# Upload the PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Check if a new file has been uploaded
if uploaded_file is not None and "uploaded_file_name" in st.session_state:
    if uploaded_file.name != st.session_state.uploaded_file_name:
        # Reset session state variables for a new upload
        st.session_state.extract = None
        st.session_state.extracted = False
        st.session_state.error_occurred = False
        st.session_state.temp_file_path = None
        st.session_state.show_skills = False
        st.session_state.show_experience = False

# Store the name of the uploaded file to detect changes
if uploaded_file is not None:
    st.session_state.uploaded_file_name = uploaded_file.name

# Initialize session state variables if they do not exist
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

def run_extraction():
    try:
        # Initialize and run the smallpdf extraction
        st.session_state.extract = smallpdf(st.session_state.temp_file_path)
        st.session_state.extract.run()
        st.session_state.extracted = True
        st.session_state.error_occurred = False
    except Exception as e:
        st.session_state.extracted = False
        st.session_state.error_occurred = True
        st.error(f"An error occurred: {e}")

if uploaded_file is not None:
    # Check if the temp file has already been created
    if st.session_state.temp_file_path is None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(uploaded_file, temp_file)
            st.session_state.temp_file_path = temp_file.name

    st.write("Thank you for uploading the PDF!")
    st.write(f"File saved at: {st.session_state.temp_file_path}")

    if st.session_state.extract is None or st.session_state.error_occurred:
        run_extraction()

    if st.session_state.extracted:
        st.write(f"Time taken: {st.session_state.extract.time}")

        # Create columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Skills"):
                st.session_state.show_skills = not st.session_state.show_skills
            if st.session_state.show_skills:
                st.write("Skills:")
                for key, values in st.session_state.extract.skills.items():
                    st.write(key)
                    st.write(','.join(values))

        with col2:
            if st.button("Experience"):
                st.session_state.show_experience = not st.session_state.show_experience
            if st.session_state.show_experience:
                st.write("Experience:")
                # Assuming 'experience' is a dictionary similar to 'skills'
                for key, values in st.session_state.extract.experience.items():
                    st.write(key)
                    for _ in key:st.write(_)

    else:
        st.error("Extraction failed. Please try again.")
        if st.button("Retry"):
            run_extraction()

