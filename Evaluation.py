#This file will hold the code that will be needed for evaluating the text sumarization algorithm.
#pip install rouge-score
#ROUGE METRIC:
from rouge_score import rouge_scorer

# Initialize ROUGE scorer for specific n-gram 
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Sample generated and reference texts
generated = "The cat sat on the mat."
reference = "The cat is sitting on the mat."

# Compute ROUGE scores
scores = scorer.score(reference, generated)

# Display the scores
print("ROUGE-1 Score:", scores['rouge1'])
print("ROUGE-2 Score:", scores['rouge2'])
print("ROUGE-L Score:", scores['rougeL'])
