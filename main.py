import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, TextTilingTokenizer
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import os
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# Extractive Summarization Functions
# -------------------------------

def read_file(file_name, encoding='utf-8'):
    """Reads a text file and splits it into sentences."""
    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found.")
        return []
    
    with open(file_name, "r", encoding=encoding) as file:
        filedata = file.read()
        sentences = sent_tokenize(filedata)
    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    """Calculates cosine similarity between two sentences."""
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stopwords]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stopwords]
    
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)


def gen_sim_matrix(sentences, stop_words):
    """Generates a sentence similarity matrix for TextRank."""
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix


def extractive_summary(file_name, top_n=5):
    """Generates an extractive summary using TextRank."""
    stop_words = stopwords.words('english')
    sentences = read_file(file_name)
    if len(sentences) == 0:
        return []
    
    similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [ranked_sentences[i][1] for i in range(min(top_n, len(ranked_sentences)))]
    
    return top_sentences

# -------------------------------
# Abstractive Summarization with PEGASUS
# -------------------------------

def abstractive_summary(text, model, tokenizer, max_length=150):
    """Generates an abstractive summary using PEGASUS."""
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = model.generate(tokens["input_ids"], max_length=max_length, num_beams=5, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -------------------------------
# Hybrid Summarization Function
# -------------------------------

def hybrid_summary(file_name, top_n=5, max_length=150):
    """Combines extractive and abstractive summarization into a hybrid approach."""
    print(f"\nProcessing file: {file_name}")
    
    # Step 1: Extractive Summarization
    print("\nPerforming extractive summarization...")
    extracted_sentences = extractive_summary(file_name, top_n)
    
    if not extracted_sentences:
        return "No content available for summarization."
    
    extractive_text = " ".join(extracted_sentences)
    print("Extractive Summary (Input to PEGASUS):\n", extractive_text)
    
    # Step 2: Abstractive Summarization
    print("\nPerforming abstractive summarization with PEGASUS...")
    abstractive_result = abstractive_summary(extractive_text, model, tokenizer, max_length)
    
    print("\nFinal Hybrid Summary:\n", abstractive_result)
    return abstractive_result

# Load PEGASUS Model
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# User Input for File Name
file_name = "test_file.txt"  # Replace with your actual file name for testing
if not os.path.exists(file_name):
    print(f"File '{file_name}' not found. Please check the file path and try again.")
else:
    hybrid_summary(file_name, top_n=5, max_length=150)


if os.path.exists(file_name):
    hybrid_summary(file_name, top_n=5, max_length=150)
else:
    print(f"File '{file_name}' not found. Please check the file path and try again.")

