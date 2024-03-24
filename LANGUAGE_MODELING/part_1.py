from collections import Counter

# Given corpus
corpus = [
    "<s> I am Sam </s>",
    "<s> Sam I am </s>",
    "<s> I am Sam </s>",
    "<s> I do not like green eggs and Sam </s>"
]

# Tokenize the corpus into words
tokens = [token for sentence in corpus for token in sentence.split()]

# Calculate bigrams
bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

# Count the bigrams and unigrams
bigram_count = Counter(bigrams)
unigram_count = Counter(tokens)

# Calculate the probability of the bigram "Sam am" using add-one smoothing
# C(am, Sam) = 2, C(am) = 3, V = 11
c_am_sam = bigram_count[('am', 'Sam')]
c_am = unigram_count['am']
vocabulary_size = len(set(tokens))

probability_sam_am = (c_am_sam + 1) / (c_am + vocabulary_size)

print("Probability of P(Sam|am): {:.2f} ".format(probability_sam_am))
