import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.cluster.util import cosine_distance

def read_file(file_name):
    file = open(file_name, "r")
    data_file = file.read()
    sentences = sent_tokenize(data_file)
    for sentence in sentences:        
        sentence.replace("[^a-zA-Z0-9]"," ") 
    return sentences

#goes through each sentence in the text file, looking for similarities between them.
def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    #helps get only unique words in the list
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))
  #creating vectors that will keep count of the amount of unique words
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
    