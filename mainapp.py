import streamlit as st
import os
import io
import json
import base64
import tempfile
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import cv2
from anthropic import Anthropic
from difflib import SequenceMatcher
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit_drawable_canvas
from streamlit_drawable_canvas import st_canvas


import sys
import importlib.util

# Run init script
spec = importlib.util.spec_from_file_location("init", "init.py")
init = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init)

# Set page configuration
st.set_page_config(
    page_title="Exam Answer Comparison Tool",
    page_icon="📝",
    layout="wide"
)

# Initialize Anthropic client
@st.cache_resource
def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    
    if api_key is None:
        st.warning("No Anthropic API key found. Please configure it in your secrets or environment variables.")
        return None
        
    return Anthropic(api_key=api_key)

# Convert PDF to images
def convert_pdf_to_images(pdf_file, dpi=300):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    try:
        images = convert_from_path(tmp_path, dpi=dpi)
        os.unlink(tmp_path)  # Delete the temporary file
        return images
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        os.unlink(tmp_path)  # Delete the temporary file
        return []

# Convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Extract text using Claude
def extract_text_with_claude(image, client):
    if client is None:
        return "Error: No Anthropic API client configured."
    
    try:
        base64_image = image_to_base64(image)
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system="You are an expert OCR system. Your task is to accurately extract text from the provided image. Extract all visible text maintaining original formatting as much as possible. If mathematical equations are present, represent them using LaTeX notation.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text", 
                            "text": "Extract all the text from this image, preserving the formatting as much as possible. If there are mathematical equations, represent them using LaTeX notation."
                        }
                    ]
                }
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        return f"Error extracting text: {e}"

# Compare extracted text with solution
def compare_text(extracted_text, solution_text):
    # Simple comparison using SequenceMatcher
    matcher = SequenceMatcher(None, extracted_text.lower(), solution_text.lower())
    similarity = matcher.ratio() * 100
    
    # Generate feedback based on similarity
    if similarity > 90:
        feedback = "Excellent match! The answer is very close to the solution."
        grade = "A"
        marks = 10
    elif similarity > 80:
        feedback = "Very good match. The answer covers most of the solution points."
        grade = "B"
        marks = 8
    elif similarity > 70:
        feedback = "Good match. The answer includes many key elements but has some differences."
        grade = "C"
        marks = 7
    elif similarity > 60:
        feedback = "Satisfactory. The answer has the right idea but is missing several key elements."
        grade = "D"
        marks = 6
    elif similarity > 40:
        feedback = "Needs improvement. The answer is somewhat related but misses many key points."
        grade = "E"
        marks = 4
    else:
        feedback = "Significant differences from the expected solution."
        grade = "F"
        marks = 2
    
    # Advanced analysis
    extracted_words = set(extracted_text.lower().split())
    solution_words = set(solution_text.lower().split())
    
    missing_keywords = solution_words - extracted_words
    extra_keywords = extracted_words - solution_words
    
    # Limit the number of keywords in feedback
    max_keywords = 5
    missing_keywords_str = ", ".join(list(missing_keywords)[:max_keywords])
    if len(missing_keywords) > max_keywords:
        missing_keywords_str += f" and {len(missing_keywords) - max_keywords} more"
    
    extra_keywords_str = ", ".join(list(extra_keywords)[:max_keywords])
    if len(extra_keywords) > max_keywords:
        extra_keywords_str += f" and {len(extra_keywords) - max_keywords} more"
    
    detailed_feedback = f"""
    Similarity score: {similarity:.2f}%
    
    Missing key terms: {missing_keywords_str if missing_keywords else "None"}
    
    Extra terms: {extra_keywords_str if extra_keywords else "None"}
    """
    
    return {
        "similarity": similarity,
        "feedback": feedback,
        "detailed_feedback": detailed_feedback,
        "grade": grade,
        "marks": marks
    }

# Get detailed feedback using Claude
def get_ai_feedback(extracted_text, solution_text, client):
    if client is None:
        return "Error: No Anthropic API client configured."
    
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            system="You are an expert teacher assistant who helps grade student answers and provide detailed, constructive feedback.",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Compare the following student answer with the teacher's solution. 
                    Provide detailed feedback including:
                    
                    1. What key concepts the student correctly identified
                    2. What important elements are missing
                    3. Any misconceptions or errors
                    4. A fair grade (A-F) and suggested marks out of 10
                    5. Specific suggestions for improvement
                    
                    Student Answer:
                    ```
                    {extracted_text}
                    ```
                    
                    Teacher's Solution:
                    ```
                    {solution_text}
                    ```
                    
                    Format your response using the headings: Key Strengths, Missing Elements, Misconceptions, Grade, and Suggestions for Improvement.
                    """
                }
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        return f"Error generating AI feedback: {e}"

# Convert PIL image to numpy array
def pil_to_numpy(pil_img):
    return np.array(pil_img)

# Convert numpy array to PIL image
def numpy_to_pil(np_img):
    return Image.fromarray(np_img)

# Crop image based on rectangle coordinates
def crop_image_from_rect(image, rect_coords):
    x_min = int(min(rect_coords[0], rect_coords[2]))
    y_min = int(min(rect_coords[1], rect_coords[3]))
    x_max = int(max(rect_coords[0], rect_coords[2]))
    y_max = int(max(rect_coords[1], rect_coords[3]))
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.width, x_max)
    y_max = min(image.height, y_max)
    
    # Create the crop
    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped

# Main Streamlit app
def main():
    st.title("Exam Answer Comparison Tool")
    
    # Initialize Anthropic client
    client = get_anthropic_client()
    
    # Initialize session state for storing uploaded images and crops
    if "pdf_images" not in st.session_state:
        st.session_state.pdf_images = []
    if "selected_image_index" not in st.session_state:
        st.session_state.selected_image_index = 0
    if "cropped_images" not in st.session_state:
        st.session_state.cropped_images = []
    if "extracted_texts" not in st.session_state:
        st.session_state.extracted_texts = []
    if "teacher_solutions" not in st.session_state:
        st.session_state.teacher_solutions = []
    if "comparisons" not in st.session_state:
        st.session_state.comparisons = []
    if "ai_feedbacks" not in st.session_state:
        st.session_state.ai_feedbacks = []
    if "crop_rect" not in st.session_state:
        st.session_state.crop_rect = None
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Files")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Converting PDF to images..."):
                    st.session_state.pdf_images = convert_pdf_to_images(uploaded_file)
                    st.session_state.selected_image_index = 0
                    st.session_state.cropped_images = []
                    st.session_state.extracted_texts = []
                    st.session_state.teacher_solutions = []
                    st.session_state.comparisons = []
                    st.session_state.ai_feedbacks = []
                    
                    # Initialize empty lists for all pages
                    st.session_state.extracted_texts = [None] * len(st.session_state.pdf_images)
                    st.session_state.teacher_solutions = [None] * len(st.session_state.pdf_images)
                    st.session_state.comparisons = [None] * len(st.session_state.pdf_images)
                    st.session_state.ai_feedbacks = [None] * len(st.session_state.pdf_images)
                st.success(f"Processed {len(st.session_state.pdf_images)} page(s)")
        
        # Page selector if PDF is loaded
        if st.session_state.pdf_images:
            st.header("Page Navigation")
            st.session_state.selected_image_index = st.selectbox(
                "Select Page",
                range(len(st.session_state.pdf_images)),
                format_func=lambda x: f"Page {x+1}",
                index=st.session_state.selected_image_index
            )
        
        # Export results button
        if st.session_state.comparisons and any(st.session_state.comparisons):
            if st.button("Export Results"):
                valid_results = []
                for i, comp in enumerate(st.session_state.comparisons):
                    if comp:
                        valid_results.append({
                            "Page": i+1,
                            "Similarity": comp["similarity"],
                            "Grade": comp["grade"],
                            "Marks": comp["marks"],
                            "Feedback": comp["feedback"],
                            "Extracted Text": st.session_state.extracted_texts[i],
                            "Teacher Solution": st.session_state.teacher_solutions[i]
                        })
                
                if valid_results:
                    df = pd.DataFrame(valid_results)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="grading_results.csv",
                        mime="text/csv"
                    )
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Image Selection", "Text Extraction", "Comparison & Feedback"])
    
    # Tab 1: Image cropping and selection
    with tab1:
        if st.session_state.pdf_images:
            current_index = st.session_state.selected_image_index
            current_image = st.session_state.pdf_images[current_index]
            
            st.subheader(f"Page {current_index + 1}")
            
            # Display the image and allow cropping
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Convert PIL image to numpy for the canvas
                img_array = pil_to_numpy(current_image)
                img_array = np.array(current_image).astype(np.uint8)
                # Create a canvas for interactive selection
                st.write("**Draw a rectangle around the answer you want to extract:**")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Orange with transparency
                    stroke_width=2,
                    stroke_color="#FF0000",  # Red border
                    background_image=img_array,
                    drawing_mode="rect",
                    key=f"canvas_{current_index}",
                    update_streamlit=True,
                    width=img_array.shape[1],
                    height=min(700, img_array.shape[0])  # Limit height for better display
                )
                
                # Process the drawn rectangle
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    # Get the last drawn rectangle
                    last_rect = canvas_result.json_data["objects"][-1]
                    
                    # Extract coordinates (scale_factor adjusts for any canvas scaling)
                    left = last_rect.get("left", 0)
                    top = last_rect.get("top", 0)
                    width = last_rect.get("width", 0)
                    height = last_rect.get("height", 0)
                    
                    # Store rectangle coordinates [x1, y1, x2, y2]
                    st.session_state.crop_rect = [left, top, left + width, top + height]
                
                # Button to extract the selected area
                if st.session_state.crop_rect and st.button("Extract Selected Area"):
                    with st.spinner("Extracting text..."):
                        # Crop the image
                        cropped_img = crop_image_from_rect(current_image, st.session_state.crop_rect)
                        
                        # Save the cropped image
                        if len(st.session_state.cropped_images) <= current_index:
                            st.session_state.cropped_images.extend([None] * (current_index + 1 - len(st.session_state.cropped_images)))
                        st.session_state.cropped_images[current_index] = cropped_img
                        
                        # Extract text using Claude
                        extracted_text = extract_text_with_claude(cropped_img, client)
                        st.session_state.extracted_texts[current_index] = extracted_text
                        
                        st.success("Image cropped and text extracted!")
            
            with col2:
                if st.session_state.cropped_images and current_index < len(st.session_state.cropped_images) and st.session_state.cropped_images[current_index] is not None:
                    st.subheader("Cropped Area")
                    st.image(st.session_state.cropped_images[current_index], use_column_width=True)
                else:
                    st.info("Draw a rectangle on the image and click 'Extract Selected Area' to crop.")
        else:
            st.info("Please upload a PDF file from the sidebar to begin.")
    
    # Tab 2: Text extraction and solution input
    with tab2:
        if st.session_state.pdf_images:
            current_index = st.session_state.selected_image_index
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Extracted Student Answer")
                if current_index < len(st.session_state.extracted_texts) and st.session_state.extracted_texts[current_index]:
                    extracted_text = st.session_state.extracted_texts[current_index]
                    # Allow editing the extracted text
                    new_extracted_text = st.text_area("Edit if needed:", value=extracted_text, height=300)
                    if new_extracted_text != extracted_text:
                        st.session_state.extracted_texts[current_index] = new_extracted_text
                else:
                    st.info("No text extracted yet. Crop an area and extract text first.")
            
            with col2:
                st.subheader("Teacher's Solution")
                # Get or update the teacher's solution
                current_solution = ""
                if current_index < len(st.session_state.teacher_solutions) and st.session_state.teacher_solutions[current_index]:
                    current_solution = st.session_state.teacher_solutions[current_index]
                
                new_solution = st.text_area("Enter or paste the model answer:", value=current_solution, height=300)
                if new_solution != current_solution:
                    if len(st.session_state.teacher_solutions) <= current_index:
                        st.session_state.teacher_solutions.extend([None] * (current_index + 1 - len(st.session_state.teacher_solutions)))
                    st.session_state.teacher_solutions[current_index] = new_solution
            
            # Compare button
            if (current_index < len(st.session_state.extracted_texts) and st.session_state.extracted_texts[current_index] and
                current_index < len(st.session_state.teacher_solutions) and st.session_state.teacher_solutions[current_index]):
                if st.button("Compare and Generate Feedback"):
                    with st.spinner("Analyzing..."):
                        # Basic comparison
                        comparison = compare_text(
                            st.session_state.extracted_texts[current_index],
                            st.session_state.teacher_solutions[current_index]
                        )
                        
                        # Store the comparison result
                        if len(st.session_state.comparisons) <= current_index:
                            st.session_state.comparisons.extend([None] * (current_index + 1 - len(st.session_state.comparisons)))
                        st.session_state.comparisons[current_index] = comparison
                        
                        # AI feedback
                        ai_feedback = get_ai_feedback(
                            st.session_state.extracted_texts[current_index],
                            st.session_state.teacher_solutions[current_index],
                            client
                        )
                        
                        # Store the AI feedback
                        if len(st.session_state.ai_feedbacks) <= current_index:
                            st.session_state.ai_feedbacks.extend([None] * (current_index + 1 - len(st.session_state.ai_feedbacks)))
                        st.session_state.ai_feedbacks[current_index] = ai_feedback
                    
                    st.success("Comparison and feedback generated!")
        else:
            st.info("Please upload a PDF file from the sidebar to begin.")
    
    # Tab 3: Comparison results and feedback
    with tab3:
        if st.session_state.pdf_images:
            current_index = st.session_state.selected_image_index
            
            if (current_index < len(st.session_state.comparisons) and 
                st.session_state.comparisons[current_index] is not None):
                
                comparison = st.session_state.comparisons[current_index]
                
                # Display comparison results
                st.subheader("Comparison Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Similarity", f"{comparison['similarity']:.2f}%")
                col2.metric("Grade", comparison["grade"])
                col3.metric("Marks", f"{comparison['marks']}/10")
                
                # Basic feedback
                st.subheader("Basic Feedback")
                st.write(comparison["feedback"])
                with st.expander("Detailed Analysis"):
                    st.write(comparison["detailed_feedback"])
                
                # AI feedback
                if (current_index < len(st.session_state.ai_feedbacks) and 
                    st.session_state.ai_feedbacks[current_index] is not None):
                    
                    st.subheader("AI-Generated Feedback")
                    st.write(st.session_state.ai_feedbacks[current_index])
                
                # Allow adjusting the marks manually
                st.subheader("Manual Grade Adjustment")
                new_marks = st.slider("Adjusted Marks", 0, 10, int(comparison["marks"]))
                new_grade = st.selectbox("Adjusted Grade", ["A", "B", "C", "D", "E", "F"], index=ord(comparison["grade"]) - ord("A"))
                
                if new_marks != comparison["marks"] or new_grade != comparison["grade"]:
                    if st.button("Update Grade"):
                        st.session_state.comparisons[current_index]["marks"] = new_marks
                        st.session_state.comparisons[current_index]["grade"] = new_grade
                        st.success("Grade updated!")
                
                # Display the compared texts side by side
                st.subheader("Comparison Side-by-Side")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Student Answer:**")
                    st.text_area("", value=st.session_state.extracted_texts[current_index], height=200, disabled=True)
                
                with col2:
                    st.markdown("**Teacher's Solution:**")
                    st.text_area("", value=st.session_state.teacher_solutions[current_index], height=200, disabled=True)
            else:
                st.info("No comparison results yet. Extract student answers and enter teacher solutions first, then click 'Compare and Generate Feedback'.")
        else:
            st.info("Please upload a PDF file from the sidebar to begin.")

if __name__ == "__main__":
    main()
