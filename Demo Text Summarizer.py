#Demo Text Summarizer
import spacy
import pytextrank
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")
example_text = 'Ever feel like you dont have enough time to read everything that you want to? What if you could run a routine that summarized documents for you, whether its your favorite news source, academic articles, or work-related documents?' \
 'Text summarization is a Natural Language Processing (NLP) task that summarizes the information in large texts for quicker consumption without losing vital information. Your favourite news aggregator (such as Google News) takes advantage of text summarization algorithms in order to provide you with information you need to know whether the article is relevant or not without having to click the link.'
#Run Spacy Pipeline with TextRank Algorithm
doc = nlp(example_text)
#Show text Summary
for sent in doc._.textrank.summary(limit_sentences=2):
    print(sent)
    
