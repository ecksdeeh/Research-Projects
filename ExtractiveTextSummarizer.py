import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, TextTilingTokenizer
from nltk import pos_tag
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Function to read file and split it into sentences
def read_file(file_name, encoding='utf-8'):
    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found.")
        return []
    
    with open(file_name, "r", encoding=encoding) as file:
        filedata = file.read()
        sentences = sent_tokenize(filedata)
    return sentences

# Function to compute similarity between two sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in word_tokenize(sent1)]
    sent2 = [w.lower() for w in word_tokenize(sent2)]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
        
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

# Function to generate similarity matrix
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

# Function to generate summary
def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    
    sentences = read_file(file_name)
    if len(sentences) == 0:
        return "No summary generated due to missing or empty file."
    
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    for i in range(min(top_n, len(ranked_sentence))):
        summarize_text.append(ranked_sentence[i][1])
    
    summary = " ".join(summarize_text)
    
    print("Summary:\n", summary)
    
    #pos_choice = input('Do you wish to see each word tagged? (Type Y/N): ').lower()
    
    #if pos_choice == 'y':
        #words = word_tokenize(" ".join(sentences))  # Tokenize entire text
        #pos_tags = pos_tag(words)
        #print("\nPOS Tagging Results:")
        #for word, pos_tagged in pos_tags:
            #print(f'{word}: {pos_tagged}')
    
    return summary

# New function to perform topic segmentation using TextTiling
def topic_segmentation(file_name):
    text_tiling_tokenizer = TextTilingTokenizer()
    
    # Read the full text file (not tokenized into sentences, but as raw text)
    with open(file_name, "r", encoding='utf-8') as file:
        raw_text = file.read()

    # Perform topic segmentation
    #If text is NOT long, skip segmentation; else: continue with segmentation
   #Specificications with the length of tokens, in order to achieve this
        segments = text_tiling_tokenizer.tokenize(raw_text)
    
    print("\nTopic Segments:")
    for idx, segment in enumerate(segments):
        print(f"\nSegment {idx + 1}:\n{segment[:200]}...")  # Display the first 200 characters of each segment
    
    return segments

# List of files to summarize and segment
file_names = ["notes.md", "BlogNotes.md", "Essay.md"]

for file_name in file_names:
    # First, perform topic segmentation
    print(f"\nPerforming topic segmentation for {file_name}:")
    segments = topic_segmentation(file_name)
    
    # Optionally summarize the most important sentences within each segment
    for idx, segment in enumerate(segments):
        print(f"\nSummary for Segment {idx + 1} in {file_name}:")
        with open(f"temp_segment_{idx}.txt", "w", encoding='utf-8') as temp_file:
            temp_file.write(segment)
        summary = generate_summary(f"temp_segment_{idx}.txt", 3)  # Top 3 sentences per segment
        print(summary)
