# Read the training corpus from the file
with open("train-Spring2024.txt", "r", encoding="utf-8") as file:
    training_corpus = file.readlines()

# Read the test corpus from the file
with open("test.txt", "r", encoding="utf-8") as file:
    test_corpus = file.readlines()

# Add start and end symbols to each sentence, and lowercase all words
training_corpus = ["<s> " + sentence.lower().strip() + " </s>" for sentence in training_corpus]
test_corpus = ["<s> " + sentence.lower().strip() + " </s>" for sentence in test_corpus]

# Tokenize the training corpus to count word occurrences
tokenized_training_corpus = [sentence.split() for sentence in training_corpus]
all_words = [word for sentence in tokenized_training_corpus for word in sentence]

# Replace words occurring once with <unk> in the training corpus
word_freq = {}
for word in all_words:
    word_freq[word] = word_freq.get(word, 0) + 1

rare_words = set([word for word, freq in word_freq.items() if freq == 1])

for i, sentence in enumerate(tokenized_training_corpus):
    tokenized_training_corpus[i] = ["<unk>" if word in rare_words else word for word in sentence]

# Replace unseen words in the test corpus with <unk>
tokenized_test_corpus = [sentence.split() for sentence in test_corpus]
for i, sentence in enumerate(tokenized_test_corpus):
    tokenized_test_corpus[i] = ["<unk>" if word not in all_words else word for word in sentence]

# Join the tokenized corpora back into sentences
training_corpus_processed = [" ".join(sentence) for sentence in tokenized_training_corpus]
test_corpus_processed = [" ".join(sentence) for sentence in tokenized_test_corpus]

# Write the processed training corpus to a new file
with open("training_processed.txt", "w", encoding="utf-8") as file:
    for sentence in training_corpus_processed:
        file.write(sentence + "\n")

# Write the processed test corpus to a new file
with open("test_processed.txt", "w", encoding="utf-8") as file:
    for sentence in test_corpus_processed:
        file.write(sentence + "\n")
