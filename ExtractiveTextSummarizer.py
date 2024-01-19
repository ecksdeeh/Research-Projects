import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
#takes file name as input, opens and reads the file, uses sent_tokenize to split text into sentences, and then returns them.
def read_file(file_name):
    file = open(file_name, "r")
    filedata = file.read()
    sentences = sent_tokenize(filedata)
    return sentences
#goes through each sentence in the text file, looking for similarities between them.
def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    #helps get only unique words in the list
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))
  #creating vectors  
    vector1 = [0] *len(all_words)
    vector2 = [0] *len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1-cosine_distance(vector1,vector2)
#generating a similarity matrix based on the similarities found in the sentences, it stores the sentences into an array(similarity matrix)
def gen_sim_matrix(sentences,stop_words):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
    return similarity_matrix
#this generates the summary by getting sentences via read_article, generating a similarity matrix via the sentences in the article, putting it inside a graph
#then using pagerank to keep score on the most relevant topics in the sentence.
#then, the sentences are ranked in order from greatest to least, then appended in that same order of importance.
#the summary is then printed.
def generate_summary(file_name,top_n=5):
    stop_words=stopwords.words('english')
    summarize_text=[]
    sentences = read_file(file_name)
    sentence_similarity_matrix = gen_sim_matrix(sentences,stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence=sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    for i in range(top_n):
        summarize_text.append("".join(ranked_sentence[i][1]))
    print("Summary \n", ". ".join(summarize_text))

generate_summary("notes.txt", 7)
    
                
