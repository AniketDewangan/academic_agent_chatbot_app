import streamlit as st
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM
)
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

# ==================== #
# ======= Setup ====== #
# ==================== #

# Set Streamlit page configuration
st.set_page_config(
    page_title="📚 Academic Research Paper Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== #
# ======= Caching ===== #
# ==================== #

@st.cache_resource
def load_per_pdf_summarizer():
    """
    Loads the Transformer-based summarization pipeline for individual PDFs.

    Returns:
        pipeline: HuggingFace summarization pipeline.
    """
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",  # Per-PDF summarization model
        tokenizer="facebook/bart-large-cnn",
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )
    return summarizer

@st.cache_resource
def load_final_summarizer():
    """
    Loads the Transformer-based summarization pipeline for final aggregation.

    Returns:
        pipeline: HuggingFace summarization pipeline.
    """
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",  # Use same model for consistency
        tokenizer="facebook/bart-large-cnn",
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )
    return summarizer

@st.cache_resource
def load_text_generator():
    """
    Loads the Transformer-based text generation pipeline.

    Returns:
        pipeline: HuggingFace text generation pipeline.
    """
    text_generator = pipeline(
        "text-generation",
        model="google/flan-t5-large",  # Efficient and free model
        tokenizer="google/flan-t5-large",
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )
    return text_generator

@st.cache_resource
def load_qa_model():
    """
    Loads the Transformer-based Question Answering pipeline.

    Returns:
        pipeline: HuggingFace question answering pipeline.
    """
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",  # Free QA model
        tokenizer="deepset/roberta-base-squad2",
        framework="pt",
        device=0 if torch.cuda.is_available() else -1
    )
    return qa_pipeline

@st.cache_resource
def load_embedding_model():
    """
    Loads the SentenceTransformer model for generating embeddings.

    Returns:
        SentenceTransformer: Pre-trained SentenceTransformer model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient
    return model

@st.cache_resource
def create_faiss_index(embeddings):
    """
    Creates a FAISS index for efficient similarity search.

    Args:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file, including captions and alt-text.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                # Extract captions and alt-text
                captions = ""
                for obj in page.objects.get('image', []):
                    if 'caption' in obj:
                        captions += obj['caption'] + "\n"
                if page_text:
                    text += page_text + "\n"
                if captions:
                    text += captions + "\n"
    except Exception as e:
        st.error(f"Failed to extract text from {pdf_path}: {e}")
    return text

def split_text_into_chunks(text, max_tokens=500, overlap=50):
    """
    Splits text into overlapping chunks based on the maximum number of tokens.

    Args:
        text (str): The text to split.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(' '.join(current_chunk))
            # Start new chunk with overlapping tokens
            current_chunk = current_chunk[-overlap:]
            current_length = len(current_chunk)

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_embeddings(text_chunks, model):
    """
    Generates embeddings for each text chunk.

    Args:
        text_chunks (list): List of text chunks.
        model (SentenceTransformer): Pre-trained SentenceTransformer model.

    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    return np.array(embeddings).astype('float32')

@st.cache_resource
def process_pdfs(pdf_dir='downloaded_papers', max_tokens=500, overlap=50):
    """
    Processes all PDFs in the specified directory.

    Args:
        pdf_dir (str): Directory containing PDF files.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        tuple: (dict mapping chunk IDs to text, FAISS index)
    """
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

    # Debugging: Check number of embeddings and FAISS index size
    st.write(f"Total Chunks Processed: {len(all_chunks)}")
    st.write(f"FAISS Index Size: {faiss_index.ntotal}")

    return chunk_id_to_text, faiss_index, chunk_id_to_paper

@st.cache_data
def load_database(db_path='database.json'):
    """
    Loads the JSON database containing stored papers.

    Args:
        db_path (str): Path to the JSON database file.

    Returns:
        dict: Database contents.
    """
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
# ==== Initialize ==== #
# ==================== #

# Load models
per_pdf_summarizer = load_per_pdf_summarizer()
final_summarizer = load_final_summarizer()
text_generator = load_text_generator()
qa_pipeline = load_qa_model()
embedding_model = load_embedding_model()

# Load and index PDFs (This will process all PDFs in the 'downloaded_papers' directory initially)
with st.spinner("Processing and indexing PDFs..."):
    chunk_id_to_text, faiss_index, chunk_id_to_paper = process_pdfs()

# Load database
database = load_database()

# ==================== #
# ======= Utils ======= #
# ==================== #

def sanitize_topic(topic):
    """
    Sanitizes the research topic to create a valid directory name.

    Args:
        topic (str): The research topic entered by the user.

    Returns:
        str: Sanitized topic string.
    """
    return "".join([c if c.isalnum() or c in " .-_()" else "_" for c in topic]).strip()

def is_within_last_n_years(published_date_str, n=5):
    """
    Checks if the published date is within the last n years.

    Args:
        published_date_str (str): Published date in ISO 8601 format.
        n (int): Number of years.

    Returns:
        bool: True if within last n years, else False.
    """
    published_date = datetime.strptime(published_date_str, "%Y-%m-%dT%H:%M:%SZ")
    n_years_ago = datetime.now() - timedelta(days=n*365)
    return published_date >= n_years_ago

def search_papers_arxiv(topic, max_results=10, n_years=5):
    """
    Searches for papers on arXiv based on the given topic and returns a list of papers with relevant details
    published within the last n years.

    Args:
        topic (str): The search query topic.
        max_results (int): The maximum number of results to retrieve. Default is 10.
        n_years (int): Number of years to look back for papers. Default is 5.

    Returns:
        list: A list of dictionaries, each containing details of a paper.
    """
    # Construct the API query URL
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"

    try:
        # Send the GET request to the arXiv API
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data from arXiv: {e}")
        return []

    # Parse the XML response
    root = ET.fromstring(response.content)

    # Define the namespace dictionary to handle XML namespaces
    namespaces = {
        'arxiv': 'http://arxiv.org/schemas/atom',
        'atom': 'http://www.w3.org/2005/Atom'
    }

    papers = []

    # Iterate over each entry in the XML feed
    for entry in root.findall('atom:entry', namespaces):
        published_elem = entry.find('atom:published', namespaces)
        if published_elem is not None and is_within_last_n_years(published_elem.text, n=n_years):
            paper = {}

            # Extract the ID
            id_elem = entry.find('atom:id', namespaces)
            paper['id'] = id_elem.text if id_elem is not None else None

            # Extract the updated date
            updated_elem = entry.find('atom:updated', namespaces)
            paper['updated'] = updated_elem.text if updated_elem is not None else None

            # Extract the published date
            paper['published'] = published_elem.text

            # Extract the title
            title_elem = entry.find('atom:title', namespaces)
            paper['title'] = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else None

            # Extract the summary
            summary_elem = entry.find('atom:summary', namespaces)
            if summary_elem is not None:
                # Clean up the summary by removing excessive whitespace
                summary = ' '.join(summary_elem.text.split())
                paper['summary'] = summary
            else:
                paper['summary'] = None

            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)
            paper['authors'] = authors

            # Extract links and find PDF URL
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
            paper['pdf_url'] = pdf_url  # Add PDF URL to paper metadata

            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                categories.append(category.attrib.get('term'))
            paper['categories'] = categories

            # Optionally extract comments (if present)
            comment_elem = entry.find('arxiv:comment', namespaces)
            paper['comment'] = comment_elem.text if comment_elem is not None else None

            papers.append(paper)

    return papers

def download_pdf(pdf_url, paper_title, published_date, topic, base_dir='downloaded_papers'):
    """
    Downloads the PDF from the given URL and saves it in a structured directory based on the research topic and publication year.

    Args:
        pdf_url (str): The URL of the PDF to download.
        paper_title (str): The title of the paper (used for naming the file).
        published_date (str): The publication date in ISO 8601 format.
        topic (str): The research topic entered by the user.
        base_dir (str): The base directory to store downloaded PDFs.

    Returns:
        str: The file path where the PDF is saved, or None if download failed.
    """
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download PDF from {pdf_url}: {e}")
        return None

    # Extract year from published_date
    year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year

    # Sanitize research topic for directory name
    sanitized_topic = sanitize_topic(topic)

    # Create topic/year directory if it doesn't exist
    topic_dir = os.path.join(base_dir, sanitized_topic, str(year))
    os.makedirs(topic_dir, exist_ok=True)

    # Sanitize paper title for filename
    sanitized_title = "".join([c if c.isalnum() or c in " .-_()" else "_" for c in paper_title])
    filename = f"{sanitized_title}.pdf"
    file_path = os.path.join(topic_dir, filename)

    # Avoid re-downloading if file already exists
    if os.path.exists(file_path):
        st.info(f"PDF already exists at {file_path}. Skipping download.")
        return file_path

    # Save PDF
    try:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        st.success(f"Downloaded PDF to {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Failed to save PDF {file_path}: {e}")
        return None

def store_papers(topic, papers, db_path='database.json', pdf_base_dir='downloaded_papers'):
    """
    Stores the fetched papers into the database under the specified topic and downloads their PDFs.

    Args:
        topic (str): The topic under which papers are categorized.
        papers (list): A list of dictionaries containing paper details.
        db_path (str): Path to the JSON file storing the database. Default is 'database.json'.
        pdf_base_dir (str): Base directory to store downloaded PDFs.

    Returns:
        None
    """
    global database  # Ensure we modify the global database

    # Get the current date
    date = datetime.now().strftime("%Y-%m-%d")

    # Initialize the topic list if it doesn't exist
    if topic not in database:
        database[topic] = []

    # Avoid storing duplicate papers based on 'id'
    existing_ids = {paper['id'] for entry in database[topic] for paper in entry['papers']}
    new_papers = [paper for paper in papers if paper['id'] not in existing_ids]

    if not new_papers:
        st.info(f"No new papers to add for topic '{topic}'.")
        return

    # Append the new papers with the current date
    for paper in new_papers:
        # Extract PDF URL consistently
        pdf_url = paper.get('links')[1]['href']
        if pdf_url:
            file_path = download_pdf(pdf_url, paper['title'], paper['published'], topic=topic, base_dir=pdf_base_dir)
            paper['pdf_path'] = file_path
            # Generate per-PDF summary
            if file_path and os.path.exists(file_path):
                text = extract_text_from_pdf(file_path)
                if text:
                    chunks = split_text_into_chunks(text, max_tokens=500, overlap=50)
                    chunk_summaries = []
                    for chunk in chunks:
                        try:
                            summary = per_pdf_summarizer(
                                chunk,
                                max_length=150,  # Adjust as needed
                                min_length=50,
                                do_sample=False,
                                truncation=True
                            )[0]['summary_text']
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

    # Save the updated database back to the file
    try:
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=4, ensure_ascii=False)
        st.success(f"Stored {len(new_papers)} new paper(s) under topic '{topic}'.")
    except Exception as e:
        st.error(f"Failed to save database: {e}")

def query_papers(topic, db_path='database.json'):
    """
    Queries and retrieves stored papers for a given topic from the database.

    Args:
        topic (str): The topic for which to retrieve papers.
        db_path (str): Path to the JSON file storing the database. Default is 'database.json'.

    Returns:
        list: A list of entries, each containing a date and a list of papers.
              Returns an empty list if the topic is not found.
    """
    # Load the database
    if not os.path.exists(db_path):
        st.warning("Database file does not exist. No papers have been stored yet.")
        return []

    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            current_database = json.load(f)
    except json.JSONDecodeError:
        st.error("Database file is corrupted. Unable to retrieve papers.")
        return []

    # Retrieve papers for the specified topic
    papers = current_database.get(topic, [])

    if not papers:
        st.warning(f"No papers found for topic '{topic}'.")

    return papers

def get_relevant_chunks_and_sources(query, model, index, chunk_id_to_text, chunk_id_to_paper, top_k=5):
    """
    Retrieves the top_k most relevant text chunks and their sources for the given query.

    Args:
        query (str): The user's question.
        model (SentenceTransformer): Pre-trained SentenceTransformer model.
        index (faiss.Index): FAISS index of embeddings.
        chunk_id_to_text (dict): Mapping from chunk ID to text.
        chunk_id_to_paper (dict): Mapping from chunk ID to paper filename.
        top_k (int): Number of top relevant chunks to retrieve.

    Returns:
        tuple: Combined relevant text, list of source info dictionaries.
    """
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

def answer_question_with_references(question, qa_pipeline, context, sources):
    """
    Answers the question using the QA pipeline and provides references.

    Args:
        question (str): The user's question.
        qa_pipeline (pipeline): HuggingFace question answering pipeline.
        context (str): Combined context from relevant chunks.
        sources (list): List of source dictionaries.

    Returns:
        tuple: Answer string, list of relevant sources used.
    """
    try:
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"]
        # Find which chunks contain the answer
        answer_sources = []
        for source in sources:
            if answer in source["text"]:
                answer_sources.append(source)
        return answer, answer_sources
    except Exception as e:
        st.error(f"An error occurred while answering the question: {e}")
        return "Could not find an answer to your question.", []

def generate_future_works(context):
    """
    Generates future research directions based on the provided context.

    Args:
        context (str): The context from which to derive future research directions.

    Returns:
        str: The generated future work suggestions.
    """
    prompt = f"Based on recent advancements, suggest future research directions for the following context:\n\n{context}"

    try:
        generated = text_generator(
            prompt,
            max_length=200,  # Adjust as needed
            min_length=50,
            do_sample=True,
            temperature=0.7
        )
        return generated[0]["generated_text"]
    except Exception as e:
        st.error(f"An error occurred while generating future work suggestions: {e}")
        return "Could not generate future work suggestions."

# ==================== #
# ======= UI Setup ===== #
# ==================== #

st.title("📚 Academic Research Paper Assistant")

st.sidebar.title("🔍 Research Topic")
topic = st.sidebar.text_input("Enter a research topic")
sanitized_topic = sanitize_topic(topic)  # Sanitize topic for directory usage
years = st.sidebar.slider("Select the number of past years to include", min_value=1, max_value=10, value=5)

if st.sidebar.button("🔎 Search Papers"):
    if topic.strip() == "":
        st.error("Please enter a valid research topic.")
    else:
        with st.spinner("Searching and fetching papers..."):
            papers = search_papers_arxiv(topic, max_results=10, n_years=years)
            if papers:
                store_papers(topic, papers)
            else:
                st.warning("No papers found or an error occurred while fetching papers.")

if st.sidebar.button("📄 Query Papers"):
    if topic.strip() == "":
        st.error("Please enter a valid research topic.")
    else:
        stored_data = query_papers(topic)
        if stored_data:
            for entry in stored_data:
                st.subheader(f"🗓️ Date: {entry['date']}")
                for idx, paper in enumerate(entry['papers'], start=1):
                    st.markdown(f"**📄 Paper {idx}: {paper['title']}**")
                    st.write(f"**🖋️ Authors:** {', '.join(paper['authors'])}")
                    st.write(f"**📅 Published:** {paper['published']}")
                    st.write(f"**📝 Summary:** {paper['summary']}")
                    # Display per-PDF summary if available
                    pdf_summary = paper.get('pdf_summary')
                    if pdf_summary:
                        st.write(f"**📝 PDF Summary:** {pdf_summary}")
                    else:
                        st.write("**📝 PDF Summary:** Not available.")

                    # Link to PDF if available
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
                        st.write("**📄 PDF:** Not available.")

                    st.markdown("---")
        else:
            st.warning(f"No papers found for topic '{topic}'.")

if st.sidebar.button("📝 Summarize Research"):
    if topic.strip() == "":
        st.error("Please enter a valid research topic.")
    else:
        stored_data = query_papers(topic)
        if stored_data:
            combined_summaries = []
            for entry in stored_data:
                for paper in entry['papers']:
                    # Use per-PDF summary if available; otherwise, use paper's summary
                    if paper.get('pdf_summary'):
                        combined_summaries.append(paper['pdf_summary'])
                    elif paper.get('summary'):
                        combined_summaries.append(paper['summary'])

            if combined_summaries:
                # Combine all summaries into one text block
                aggregated_text = " ".join(combined_summaries)

                # Define the maximum tokens per chunk (500 for bart-large-cnn)
                max_tokens = 500
                overlap = 50

                # Split the aggregated text into overlapping chunks
                chunks = split_text_into_chunks(aggregated_text, max_tokens=max_tokens, overlap=overlap)

                # Summarize each chunk using the final summarizer
                chunk_summaries = []
                for idx, chunk in enumerate(chunks, start=1):
                    try:
                        with st.spinner(f"Summarizing chunk {idx}/{len(chunks)}..."):
                            summary = final_summarizer(
                                chunk,
                                max_length=150,  # Increased for more detailed summaries
                                min_length=50,
                                do_sample=False,
                                truncation=True
                            )[0]['summary_text']
                            chunk_summaries.append(summary)
                    except Exception as e:
                        st.error(f"Error summarizing chunk {idx}: {e}")

                # Combine all chunk summaries into one text block
                final_aggregated_summary = " ".join(chunk_summaries)

                # Check if the final aggregated summary is within model's max length
                final_max_length = final_summarizer.tokenizer.model_max_length
                final_encoded = final_summarizer.tokenizer.encode(final_aggregated_summary, truncation=True, max_length=final_max_length)
                if len(final_encoded) >= final_max_length:
                    st.warning("Final aggregated summary is too long and has been truncated for final summarization.")
                    final_aggregated_summary = final_summarizer.tokenizer.decode(final_encoded, skip_special_tokens=True)

                # Generate the comprehensive summary
                try:
                    with st.spinner("Generating comprehensive summary..."):
                        overall_summary = final_summarizer(
                            final_aggregated_summary,
                            max_length=300,  # Adjust as needed
                            min_length=100,
                            do_sample=False,
                            truncation=True
                        )[0]['summary_text']

                    # Display the overall summary
                    st.subheader("📝 Comprehensive Summary of Research Papers:")
                    st.write(overall_summary)
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")
            else:
                st.warning("No summaries available for the stored papers.")
        else:
            st.warning(f"No papers found for topic '{topic}'.")

st.markdown("---")

# ==================== #
# === Q/A Chatbot ===== #
# ==================== #

st.header("💬 Real-time Q/A Chatbot")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

chat_container = st.container()

with chat_container:
    for chat in st.session_state['chat_history']:
        if chat['is_user']:
            st.markdown(f"**You:** {chat['message']}")
        else:
            st.markdown(f"**Bot:** {chat['message']}")

    user_input = st.text_input("Ask a question about your PDFs", key="input")

    if st.button("Send"):
        if user_input.strip() == "":
            st.error("Please enter a valid question.")
        else:
            # Display user message
            st.session_state['chat_history'].append({"is_user": True, "message": user_input})
            with st.spinner("Generating answer..."):
                # Define the PDF directory based on the current topic
                if topic.strip() != "":
                    sanitized_topic = sanitize_topic(topic)
                    pdf_dir = os.path.join('downloaded_papers', sanitized_topic)
                    chunk_id_to_text, faiss_index, chunk_id_to_paper = process_pdfs(pdf_dir=pdf_dir)
                else:
                    chunk_id_to_text, faiss_index, chunk_id_to_paper = {}, None, {}

                if faiss_index and chunk_id_to_text:
                    context, sources = get_relevant_chunks_and_sources(
                        user_input, embedding_model, faiss_index, chunk_id_to_text, chunk_id_to_paper
                    )
                    answer, answer_sources = answer_question_with_references(
                        user_input, qa_pipeline, context, sources
                    )
                    # Display bot response
                    st.session_state['chat_history'].append({"is_user": False, "message": answer})
                    # Display sources
                    if answer_sources:
                        st.markdown("**Sources:**")
                        for src in answer_sources:
                            st.markdown(f"- **Paper:** {src['paper']}, **Chunk ID:** {src['chunk_id']}")
                            st.markdown(f"  > {src['text']}")
                    else:
                        st.markdown("**No specific sources found for the answer.**")
                else:
                    answer = "No PDF data available to answer your question."
                    st.session_state['chat_history'].append({"is_user": False, "message": answer})
            # Refresh the chat
            st.rerun()

st.markdown("---")

# ==================== #
# ==== Future Work ==== #
# ==================== #

st.header("🔮 Generate Future Work Suggestions")
if st.button("💡 Generate Suggestions"):
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
                st.subheader("🔮 Future Work Suggestions:")
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
