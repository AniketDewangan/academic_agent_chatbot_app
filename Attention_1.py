import streamlit as st
from transformers import pipeline
from datetime import datetime, timedelta
import requests
import json
import xml.etree.ElementTree as ET
import os
import torch
import math
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# ==================== #
# ======= Setup ====== #
# ==================== #

# Load environment variables from .env file (if using)
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Academic Research Paper Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure API key for Gemini
GENIUS_API_KEY = os.getenv('GENIUS_API_KEY', 'AIzaSyA-Qj4rxmI8mS6JbGwBpQayinnlAsS4iQA')  # Replace with your actual API key
genai.configure(api_key=GENIUS_API_KEY)

# Create GenerativeModel instance
model = genai.GenerativeModel('gemini-1.5-flash')

# ==================== #
# ======= Caching ===== #
# ==================== #

def chat_with_gemini(prompt, model, max_tokens=500):
    if not model:
        return "Gemini model is not loaded properly."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while interacting with Gemini: {e}")
        return "Could not generate a response at this time."

@st.cache_resource
def load_text_generator():
    text_generator = pipeline(
        "text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )
    return text_generator

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_resource
def load_summarizer_with_gemini():
    return summarize_with_gemini

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Failed to extract text from {pdf_path}: {e}")
    return text

def split_text_into_chunks(text, max_tokens=500, overlap=50):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = len(current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return np.array(embeddings).astype('float32')

@st.cache_resource
def process_pdfs(pdf_dir='downloaded_papers', max_tokens=500, overlap=50):
    embedding_model = load_embedding_model()
    all_chunks = []
    chunk_id_to_text = {}
    chunk_id_to_paper = {}
    chunk_id = 0
    for root, dirs, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                chunks = split_text_into_chunks(text, max_tokens=max_tokens, overlap=overlap)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_id_to_text[chunk_id] = chunk
                    chunk_id_to_paper[chunk_id] = file
                    chunk_id += 1

    if not all_chunks:
        st.warning(f"No PDF files found in '{pdf_dir}'. Please add PDFs to proceed.")
        return {}, None, {}

    embeddings = get_embeddings(all_chunks, embedding_model)
    if embeddings.size == 0:
        st.warning("No text extracted from PDFs. Ensure PDFs are not empty or corrupted.")
        return {}, None, {}

    faiss_index = create_faiss_index(embeddings)
    return chunk_id_to_text, faiss_index, chunk_id_to_paper

@st.cache_data
def load_database(db_path='database.json'):
    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                st.error("Database file is corrupted. Starting with an empty database.")
                return {}
    else:
        return {}

# ==================== #
# ======= Utils ======= #
# ==================== #

def sanitize_topic(topic):
    return "".join([c if c.isalnum() or c in " .-_()" else "_" for c in topic]).strip()

def is_within_last_n_years(published_date_str, n=5):
    published_date = datetime.strptime(published_date_str, "%Y-%m-%dT%H:%M:%SZ")
    n_years_ago = datetime.now() - timedelta(days=n*365)
    return published_date >= n_years_ago

def search_papers_arxiv(topic, max_results=10, n_years=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data from arXiv: {e}")
        return []

    root = ET.fromstring(response.content)
    namespaces = {'arxiv': 'http://arxiv.org/schemas/atom', 'atom': 'http://www.w3.org/2005/Atom'}
    papers = []

    for entry in root.findall('atom:entry', namespaces):
        published_elem = entry.find('atom:published', namespaces)
        if published_elem is not None and is_within_last_n_years(published_elem.text, n=n_years):
            paper = {}
            id_elem = entry.find('atom:id', namespaces)
            paper['id'] = id_elem.text if id_elem is not None else None
            updated_elem = entry.find('atom:updated', namespaces)
            paper['updated'] = updated_elem.text if updated_elem is not None else None
            paper['published'] = published_elem.text
            title_elem = entry.find('atom:title', namespaces)
            paper['title'] = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else None
            summary_elem = entry.find('atom:summary', namespaces)
            if summary_elem is not None:
                summary = ' '.join(summary_elem.text.split())
                paper['summary'] = summary
            else:
                paper['summary'] = None
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)
            paper['authors'] = authors
            links = []
            pdf_url = None
            for link in entry.findall('atom:link', namespaces):
                rel = link.attrib.get('rel')
                href = link.attrib.get('href')
                if rel == 'alternate' and href.endswith('.pdf'):
                    pdf_url = href
                link_info = {
                    'href': href,
                    'rel': rel,
                    'type': link.attrib.get('type'),
                    'title': link.attrib.get('title', '')
                }
                links.append(link_info)
            paper['links'] = links
            paper['pdf_url'] = pdf_url
            categories = []
            for category in entry.findall('atom:category', namespaces):
                categories.append(category.attrib.get('term'))
            paper['categories'] = categories
            comment_elem = entry.find('arxiv:comment', namespaces)
            paper['comment'] = comment_elem.text if comment_elem is not None else None
            papers.append(paper)

    return papers

def download_pdf(pdf_url, paper_title, published_date, topic, base_dir='downloaded_papers'):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download PDF from {pdf_url}: {e}")
        return None

    year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year
    sanitized_topic = sanitize_topic(topic)
    topic_dir = os.path.join(base_dir, sanitized_topic, str(year))
    os.makedirs(topic_dir, exist_ok=True)
    sanitized_title = "".join([c if c.isalnum() or c in " .-_()" else "_" for c in paper_title])
    filename = f"{sanitized_title}.pdf"
    file_path = os.path.join(topic_dir, filename)

    if os.path.exists(file_path):
        st.info(f"PDF already exists at {file_path}. Skipping download.")
        return file_path

    try:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        st.success(f"Downloaded PDF to {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Failed to save PDF {file_path}: {e}")
        return None

def store_papers(topic, papers, db_path='database.json', pdf_base_dir='downloaded_papers'):
    global database

    date = datetime.now().strftime("%Y-%m-%d")

    if topic not in database:
        database[topic] = []

    existing_ids = {paper['id'] for entry in database[topic] for paper in entry['papers']}
    new_papers = [paper for paper in papers if paper['id'] not in existing_ids]

    if not new_papers:
        st.info(f"No new papers to add for topic '{topic}'.")
        return

    for paper in new_papers:
        pdf_url = paper.get('links')[1].get('href') if len(paper.get('links', [])) > 1 else None

        if pdf_url:
            file_path = download_pdf(pdf_url, paper['title'], paper['published'], topic=topic, base_dir=pdf_base_dir)
            paper['pdf_path'] = file_path
            if file_path and os.path.exists(file_path):
                text = extract_text_from_pdf(file_path)
                if text:
                    chunks = split_text_into_chunks(text, max_tokens=500, overlap=50)
                    chunk_summaries = []
                    for chunk in chunks:
                        try:
                            summary = summarize_with_gemini(chunk, model, max_tokens=150)
                            chunk_summaries.append(summary)
                        except Exception as e:
                            st.error(f"Error summarizing PDF '{paper['title']}': {e}")
                    if chunk_summaries:
                        paper['pdf_summary'] = " ".join(chunk_summaries)
                    else:
                        paper['pdf_summary'] = None
                else:
                    paper['pdf_summary'] = None
            else:
                paper['pdf_summary'] = None
        else:
            paper['pdf_path'] = None
            paper['pdf_summary'] = None

    database[topic].append({"date": date, "papers": new_papers})

    try:
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=4, ensure_ascii=False)
        st.success(f"Stored {len(new_papers)} new paper(s) under topic '{topic}'.")
    except Exception as e:
        st.error(f"Failed to save database: {e}")

def query_papers(topic, db_path='database.json'):
    if not os.path.exists(db_path):
        st.warning("Database file does not exist. No papers have been stored yet.")
        return []

    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            current_database = json.load(f)
    except json.JSONDecodeError:
        st.error("Database file is corrupted. Unable to retrieve papers.")
        return []

    papers = current_database.get(topic, [])

    if not papers:
        st.warning(f"No papers found for topic '{topic}'.")

    return papers

def get_relevant_chunks_and_sources(query, model, index, chunk_id_to_text, chunk_id_to_paper, top_k=5):
    if index is None or not chunk_id_to_text:
        st.warning("FAISS index or chunk mapping is not available.")
        return "", []

    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    relevant_text = ""
    sources = []
    for idx in indices[0]:
        chunk_text = chunk_id_to_text.get(idx, "")
        paper_name = chunk_id_to_paper.get(idx, "Unknown Paper")
        relevant_text += chunk_text + " "
        sources.append({"paper": paper_name, "chunk_id": idx, "text": chunk_text})
    return relevant_text.strip(), sources

def answer_question_with_references(question, context, sources, model):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    answer = chat_with_gemini(prompt, model)
    return answer

def generate_future_works(context):
    prompt = f"Based on recent advancements, suggest future research directions for the following context:\n\n{context}"

    try:
        generated = text_generator(
            prompt,
            max_length=200,
            min_length=50,
            do_sample=True,
            temperature=0.7
        )
        return generated[0]["generated_text"]
    except Exception as e:
        st.error(f"An error occurred while generating future work suggestions: {e}")
        return "Could not generate future work suggestions."

def summarize_with_gemini(text, model, max_tokens=150):
    """
    Summarizes the given text using the Gemini model.

    Args:
        text (str): The text to be summarized.
        model: The configured Gemini model instance.
        max_tokens (int): Maximum number of tokens for the summary.

    Returns:
        str: The summarized text.
    """
    if not model:
        return "Gemini model is not loaded properly."

    prompt = text

    try:
        response = chat_with_gemini(prompt, model)
        summary = response
        return summary
    except Exception as e:
        st.error(f"An error occurred during summarization with Gemini: {e}")
        return "Could not generate a summary at this time."

# ==================== #
# ==== Initialize ==== #
# ==================== #

# Load models
text_generator = load_text_generator()
embedding_model = load_embedding_model()

# Load summarizer with Gemini
summarize_with_gemini = load_summarizer_with_gemini()

# Load database
database = load_database()

# ==================== #
# ======= UI Setup ===== #
# ==================== #

st.title("Academic Research Paper Assistant")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", ["Home", "Search Papers", "Query Papers", "Summarize Research", "Q/A Chatbot", "Future Work"])

# Common Variables
if 'topic' not in st.session_state:
    st.session_state['topic'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if app_mode == "Home":
    st.header("Welcome to the Academic Research Paper Assistant!")

    st.markdown("""
    This assistant helps you to:
    - Search and download research papers from arXiv.
    - Summarize multiple research papers.
    - Interact with a chatbot trained on your PDFs.
    - Generate future research directions.
    """)

elif app_mode == "Search Papers":
    st.header("Search Papers")
    st.session_state['topic'] = st.text_input("Enter a research topic", key='search_topic')
    years = st.slider("Select the number of past years to include", min_value=1, max_value=10, value=5)

    if st.button("Search and Download"):
        topic = st.session_state['topic']
        if topic.strip() == "":
            st.error("Please enter a valid research topic.")
        else:
            with st.spinner("Searching and fetching papers..."):
                papers = search_papers_arxiv(topic, max_results=10, n_years=years)
                if papers:
                    store_papers(topic, papers)
                else:
                    st.warning("No papers found or an error occurred while fetching papers.")

elif app_mode == "Query Papers":
    st.header("Query Papers")
    topic = st.text_input("Enter a research topic to query", key='query_topic')

    if st.button("Query"):
        if topic.strip() == "":
            st.error("Please enter a valid research topic.")
        else:
            stored_data = query_papers(topic)
            if stored_data:
                for entry in stored_data:
                    st.subheader(f"Date: {entry['date']}")
                    for idx, paper in enumerate(entry['papers'], start=1):
                        st.markdown(f"**Paper {idx}: {paper['title']}**")
                        st.write(f"**Authors:** {', '.join(paper['authors'])}")
                        st.write(f"**Published:** {paper['published']}")
                        st.write(f"**Summary:** {paper['summary']}")
                        pdf_summary = paper.get('pdf_summary')
                        if pdf_summary:
                            st.write(f"**PDF Summary:** {pdf_summary}")
                        else:
                            st.write("**PDF Summary:** Not available.")
                        pdf_path = paper.get('pdf_path')
                        if pdf_path and os.path.exists(pdf_path):
                            pdf_filename = os.path.basename(pdf_path)
                            try:
                                with open(pdf_path, "rb") as pdf_file:
                                    PDFbyte = pdf_file.read()
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=PDFbyte,
                                    file_name=pdf_filename,
                                    mime='application/pdf'
                                )
                            except Exception as e:
                                st.error(f"Failed to read PDF {pdf_filename}: {e}")
                        else:
                            st.write("**PDF:** Not available.")
                        st.markdown("---")
            else:
                st.warning(f"No papers found for topic '{topic}'.")

elif app_mode == "Summarize Research":
    st.header("Summarize Research")
    topic = st.text_input("Enter a research topic to summarize", key='summarize_topic')

    if st.button("Summarize"):
        if topic.strip() == "":
            st.error("Please enter a valid topic.")
        else:
            with st.spinner("Generating summary ..."):
                sanitized_topic = sanitize_topic(topic)
                pdf_dir = os.path.join('downloaded_papers', sanitized_topic)
                all_text = ""
                for root, dirs, files in os.walk(pdf_dir):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(root, file)
                            text = extract_text_from_pdf(pdf_path)
                            all_text += text + "\n"

                if all_text.strip() != "":
                    # Use Gemini for summarization
                    summary = summarize_with_gemini('summarize all text ' + all_text, model, max_tokens=500)
                    if summary:
                        st.subheader("Summary:")
                        st.write(summary)
                    else:
                        st.error("Failed to generate summary.")
                else:
                    st.error("No text extracted from PDFs to summarize.")

elif app_mode == "Q/A Chatbot":
    st.header("Real-time Q/A Chatbot")

    user_input = st.text_input("Ask a question about your PDFs", key="chat_input")

    if st.button("Send"):
        if user_input.strip() == "":
            st.error("Please enter a valid question.")
        else:
            st.session_state['chat_history'].append({"is_user": True, "message": user_input})
            with st.spinner("Generating answer..."):
                
                sanitized_topic = sanitize_topic(st.session_state['topic'])
                pdf_dir = os.path.join('downloaded_papers', sanitized_topic)
                all_text = ""
                for root, dirs, files in os.walk(pdf_dir):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(root, file)
                            text = extract_text_from_pdf(pdf_path)
                            all_text += text + "\n"
    
                #print(all.strip())
                if all_text.strip() != "":
                    prompt = f"Question: {user_input}\n answer n a single line from  Context: {all_text}\nAnswer:"
                    answer = summarize_with_gemini(prompt, model)
                    # Assuming the answer starts with "Answer:" or similar, you may need to process it
                    #answer = answer[23:] if len(answer) > 23 else answer
                    st.session_state['chat_history'].append({"is_user": False, "message": answer})
    
                else:
                    st.error("No text extracted from PDFs to answer the question.")

    if st.session_state['chat_history']:
        st.markdown("### Chat History")
        for chat in st.session_state['chat_history']:
            if chat['is_user']:
                st.markdown(f"**You:** {chat['message']}")
            else:
                st.markdown(f"**Bot:** {chat['message']}")

elif app_mode == "Future Work":
    st.header("Generate Future Work Suggestions")
    topic = st.text_input("Enter a research topic for future work suggestions", key='future_topic')

    if st.button("Generate Suggestions"):
        if topic.strip() == "":
            st.error("Please enter a valid research topic.")
        else:
            stored_data = query_papers(topic)
            if stored_data:
                summaries = [paper['summary'] for entry in stored_data for paper in entry['papers'] if paper.get('summary')]
                limited_summaries = summaries[-5:] if len(summaries) > 5 else summaries
                context = " ".join(limited_summaries)
                if context:
                    with st.spinner("Generating future work suggestions..."):
                        future_works = generate_future_works(context)
                    st.subheader("Future Work Suggestions:")
                    st.write(future_works)
                else:
                    st.warning("No summaries available to generate context for future work suggestions.")
            else:
                st.warning(f"No papers found for topic '{topic}'.")

# ==================== #
# === Footer Section ===#
# ==================== #
st.markdown("---")
st.markdown("© 2024 Academic Research Paper Assistant. All rights reserved.")
