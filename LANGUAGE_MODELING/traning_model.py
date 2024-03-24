from collections import Counter
import math

# Read the training corpus
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split()

train_corpus = read_corpus("train-Spring2024.txt")

# Calculate unigram and bigram counts
unigram_counts = Counter(train_corpus)
bigram_counts = Counter(zip(train_corpus, train_corpus[1:]))

# Train unigram maximum likelihood model
total_tokens = sum(unigram_counts.values())
unigram_mle = {word: count / total_tokens for word, count in unigram_counts.items()}

# Train bigram maximum likelihood model
bigram_mle = {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}

# Train bigram model with Add-One smoothing
vocab_size = len(unigram_counts)
bigram_add_one = {(word1, word2): (count + 1) / (unigram_counts[word1] + vocab_size) for (word1, word2), count in bigram_counts.items()}

# Example usage
print("Unigram MLE probability of 'the':", unigram_mle['the'])
print("Bigram MLE probability of ('the', 'cat'):", bigram_mle[('the', 'cat')])
print("Bigram Add-One probability of ('the', 'cat'):", bigram_add_one[('the', 'cat')])
