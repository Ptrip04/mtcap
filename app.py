
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import json
from sklearn.datasets import fetch_20newsgroups

# --- CONFIGURATION ---
# Load environment variables from a .env file (if it exists)
load_dotenv()

# Configure the Streamlit page
st.set_page_config(
    page_title="Intelligent Document Extractor",
    page_icon="📄",
    layout="wide"
)

# Configure the Google Gemini Pro Vision API
# The API key is fetched from the environment variables, which we'll set in the Colab environment later.
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("🚨 Google API Key not found! Please set it in your environment.")
    st.stop()
except Exception as e:
    st.error(f"🚨 An error occurred during Gemini configuration: {e}")
    st.stop()

# --- DATASET LOADING ---
# Load the 20 newsgroups dataset when the app starts
# Store it in Streamlit's session state to avoid reloading on each interaction
@st.cache_resource # Use cache_resource for large objects like datasets
def load_newsgroups_dataset():
    try:
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        st.success("✅ 20 Newsgroups dataset loaded successfully!")
        return newsgroups
    except Exception as e:
        st.error(f"🚨 Failed to load 20 Newsgroups dataset: {e}")
        return None

if 'newsgroups_data' not in st.session_state:
    st.session_state.newsgroups_data = load_newsgroups_dataset()
    # Initialize selected_category and selected_doc_original_index when data is loaded
    if st.session_state.newsgroups_data:
        st.session_state.selected_category = st.session_state.newsgroups_data.target_names[0]
        # Find the index of the first document in the initial category
        initial_category_index = st.session_state.newsgroups_data.target_names.index(st.session_state.selected_category)
        initial_docs_indices = [i for i, target in enumerate(st.session_state.newsgroups_data.target) if target == initial_category_index]
        if initial_docs_indices:
            st.session_state.selected_doc_original_index = initial_docs_indices[0]
        else:
            st.session_state.selected_doc_original_index = None


# --- MODEL AND HELPER FUNCTIONS ---

def get_gemini_response(input_prompt, documents):
    """
    Function to get a response from the Gemini Pro Vision model.
    :param input_prompt: The natural language instruction from the user.
    :param documents: A list of document parts (images) to be analyzed.
    :return: The model's response text.
    """
    # Update to use a currently available model, like gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Combine the system instruction, user prompt, and documents
    full_prompt = [input_prompt] + documents
    try:
        response = model.generate_content(full_prompt, stream=False)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return None

def pdf_to_images(uploaded_file):
    """
    Converts an uploaded PDF file into a list of PIL Image objects.
    This fulfills SRS Requirement 3.1.1 (Document Input Handling for PDFs).
    """
    images = []
    try:
        # Open the PDF file from the uploaded bytes
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Render page to an image (pixmap)
            pix = page.get_pixmap()
            # Convert pixmap to a PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        pdf_document.close()
    except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return []
    return images

def prepare_documents_for_model(uploaded_files):
    """
    Prepares uploaded files (PDFs or images) for the Gemini model.
    This supports SRS Requirement 3.1.1 and 3.1.5 (Multi-Modal Analysis).
    """
    document_parts = []
    if not uploaded_files:
        return document_parts

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            # Handles multi-page PDFs, addressing SRS Req 3.1.4 (Tabular Data Extraction across pages)
            images = pdf_to_images(uploaded_file)
            for img in images:
                # Convert PIL Image to a format compatible with the model (e.g., bytes or specific structure)
                # For gemini-1.5-flash, simply passing the PIL image object directly often works.
                document_parts.append(img)
        elif file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
            # Handles image files, including scans and handwritten notes (SRS Req 3.1.6)
            img = Image.open(uploaded_file)
            # Convert PIL Image to a format compatible with the model
            document_parts.append(img)
    return document_parts


# --- STREAMLIT UI ---

st.title("📄 Intelligent Document Extraction Powered by GenAI")
st.markdown(
    """
    **This application demonstrates the capabilities outlined in the SRS document.**
    Upload a document (PDF, image, handwritten note), specify what you need in natural language, and let GenAI do the work.
    """
)

# Sidebar for controls (fulfills SRS Req 3.5 Usability)
with st.sidebar:
    st.header("⚙️ Controls")

    # SRS Requirement 3.1.1: Document Input Handling
    uploaded_files = st.file_uploader(
        "1. Upload Your Documents",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload contracts, blueprints, handwritten notes, etc."
    )

    # SRS Requirement 3.1.2: Natural Language Input for Extraction
    user_prompt = st.text_area(
        "2. Specify What to Extract",
        height=150,
        placeholder="e.g., Extract the effective date, party names, termination clause, and all payment amounts from the contract.",
        help="Describe the fields you want in plain English."
    )

    # Submit button
    submit_button = st.button("🚀 Extract Data", type="primary")

    st.markdown("---")
    st.header("📚 Explore Dataset")

    # Add dataset interaction components if data is loaded
    if st.session_state.newsgroups_data:
        # Select category
        # Use the session state value as the default if it exists
        selected_category = st.selectbox(
            "Select a Category",
            st.session_state.newsgroups_data.target_names,
            index=st.session_state.newsgroups_data.target_names.index(st.session_state.selected_category) if 'selected_category' in st.session_state else 0,
            key='dataset_category_selectbox' # Add a unique key
        )
        # Update session state when selectbox value changes
        st.session_state.selected_category = selected_category


        # Filter documents by selected category
        category_index = st.session_state.newsgroups_data.target_names.index(st.session_state.selected_category)
        category_docs_indices = [i for i, target in enumerate(st.session_state.newsgroups_data.target) if target == category_index]
        category_docs_preview = [f"Document {i+1} (Index: {idx})" for i, idx in enumerate(category_docs_indices)]

        # Select a specific document from the filtered list
        # Find the index of the currently selected document within the new category's document list
        current_selected_doc_index_in_category = 0
        if 'selected_doc_original_index' in st.session_state and st.session_state.selected_doc_original_index in category_docs_indices:
             current_selected_doc_index_in_category = category_docs_indices.index(st.session_state.selected_doc_original_index)
        elif category_docs_indices:
             # If the previously selected document is not in the new category, default to the first document
             st.session_state.selected_doc_original_index = category_docs_indices[0]
             current_selected_doc_index_in_category = 0
        else:
            st.session_state.selected_doc_original_index = None # Handle case with no documents in category


        selected_doc_preview = st.selectbox(
            "Select a Document",
            category_docs_preview,
             index=current_selected_doc_index_in_category, # Set default to the document currently in session state
             key='dataset_document_selectbox' # Add a unique key
        )

        # Extract the original index from the preview string and store it in session state
        if selected_doc_preview:
             st.session_state.selected_doc_original_index = int(selected_doc_preview.split('(Index: ')[1].replace(')', ''))
        else:
             st.session_state.selected_doc_original_index = None # Handle case with no documents in category


        # Button to display the selected document
        display_doc_button = st.button("📖 Display Selected Document", key='display_doc_button')

# --- Main Content Area ---
if submit_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one document.")
    elif not user_prompt:
        st.warning("⚠️ Please specify what you want to extract.")
    else:
        with st.spinner("Analyzing document(s) and extracting data... This may take a moment."):
            # Prepare documents for the model
            document_parts = prepare_documents_for_model(uploaded_files)

            if document_parts:
                # System prompt to guide the model's behavior (SRS Req 3.1.3 & 3.1.9)
                # This prompt explicitly asks for JSON, fulfilling the structured output requirement.
                input_prompt = f"""
                You are a world-class document extraction expert. Your task is to analyze the provided document(s) and extract the information requested by the user.

                **User's Request:** "{user_prompt}"

                **Instructions:**
                1. Carefully analyze all pages/images of the document.
                2. Extract the data exactly as requested.
                3. If the request involves tables (especially ones split across pages, per SRS Req 3.1.4), consolidate them into a single coherent structure.
                4. For legal-specific requests (clauses, risks, summaries per SRS Req 3.1.7), provide the text of the clause, a brief risk assessment if applicable, or a summary as requested.
                5. Structure your final output as a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting like ```json ... ``` before or after the JSON. The output must be parsable.
                6. If a piece of information cannot be found, represent its value as `null` in the JSON.
                """

                # Get the response from the Gemini model
                response_text = get_gemini_response(input_prompt, document_parts)

                if response_text:
                    st.subheader("✅ Extraction Complete")
                    st.markdown("---")

                    # Display the results
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.info("Extracted Data (JSON)")
                        # SRS Requirement 3.1.9: Output Generation (JSON)
                        try:
                            # Attempt to parse and display as clean JSON
                            json_output = json.loads(response_text)
                            st.json(json_output)
                        except json.JSONDecodeError:
                            # If parsing fails, show the raw text for debugging
                            st.warning("Model output was not valid JSON. Displaying raw text:")
                            st.text_area("Raw Output", response_text, height=400)

                    with col2:
                        st.info("Document Preview")
                        # Show thumbnails of the first few pages/images
                        for doc in document_parts[:5]: # Show max 5 previews
                            st.image(doc, use_column_width=True)
                        if len(document_parts) > 5:
                            st.write(f"... and {len(document_parts) - 5} more pages/images.")

                    # SRS Requirement 3.1.9: Output Generation (Downloadable)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=response_text,
                        file_name="extracted_data.json",
                        mime="application/json",
                    )
                else:
                    st.error("Failed to get a response from the extraction model.")

# Display selected dataset document
# Ensure selected_doc_original_index and selected_category exist before accessing them
if 'display_doc_button' in st.session_state and st.session_state.display_doc_button and st.session_state.newsgroups_data:
    if 'selected_doc_original_index' in st.session_state and st.session_state.selected_doc_original_index is not None and 'selected_category' in st.session_state:
        st.subheader("📖 Selected Dataset Document")
        st.text_area(
            f"Content of Document (Index: {st.session_state.selected_doc_original_index}) from Category: {st.session_state.selected_category}",
            st.session_state.newsgroups_data.data[st.session_state.selected_doc_original_index],
            height=600
        )
    else:
        st.warning("Please select a category and a document from the sidebar.")
