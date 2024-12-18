from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Load the PEGASUS model and tokenizer
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

#need to adjust the algorithm to be able to accept markdown files
file_name = '''Random Forest is a type of ensemble machine learning algorithm called bagging. It is a popular variation of bagged decision trees. 
A decision tree is a branched model that consists of a hierarchy of decision nodes, where each decision node splits the data based on a decision rule. Training a decision tree involves a greedy selection of the best split points (i.e., points that divide the input space best) by minimizing a cost function. 
The greedy approach through which decision trees construct their decision boundaries makes them susceptible to high variance. This means that small changes in the training dataset can lead to very different tree structures and, in turn, model predictions. If the decision tree is not pruned, it will also tend to capture noise and outliers in the training data. This sensitivity to the training data makes decision trees susceptible to overfitting. 
Bagged decision trees address this susceptibility by combining the predictions from multiple decision trees, each trained on a bootstrap sample of the training dataset created by sampling the dataset with replacement. The limitation of this approach stems from the fact that the same greedy approach trains each tree, and some samples may be picked several times during training, making it very possible that the trees share similar (or the same) split points (hence, resulting in correlated trees). 
The Random Forest algorithm tries to mitigate this correlation by training each tree on a random subset of the training data, created by randomly sampling the dataset without replace '''
text = file_name
# Tokenize the input
tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")

# Generate summary; Parameters used: Max_length: controls maximum summary length, num_beams = number of beams searched for, higher values usually lead to better summaries
#early_stopping stops generation early if beams reach to the end, length penalty depends on the length of the summary.
summary_ids = model.generate(tokens["input_ids"], max_length=150, num_beams=20, length_penalty=3.0, early_stopping=False)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("PEGASUS Summary:", summary)
