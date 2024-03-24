from collections import Counter
import math
from itertools import islice

def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split()

def calculate_word_types_and_tokens(corpus, exclude_token='<s>'):
    count = Counter(corpus)
    num_types = len(count)
    num_tokens = sum(count.values()) - count[exclude_token]
    return num_types, num_tokens

def calculate_unseen_percentage(test_corpus, train_vocab):
    test_count = Counter(test_corpus)
    percentage_unk_types = (test_count["<unk>"] - 1) / (len(test_count) - 1) * 100
    percentage_unk_tokens = test_count["<unk>"] / (sum(test_count.values()) - test_count["<s>"]) * 100
    return percentage_unk_types, percentage_unk_tokens

def calculate_unseen_bigrams(test_corpus, train_vocab):
    test_corpus_unk = ["<unk>" if word not in train_vocab else word for word in test_corpus]
    test_bigrams = list(zip(test_corpus_unk, test_corpus_unk[1:]))
    test_bigram_count = Counter(test_bigrams)

    train_corpus_unk = ["<unk>" if word not in train_vocab else word for word in train_corpus]
    train_bigrams = list(zip(train_corpus_unk, train_corpus_unk[1:]))
    train_bigram_count = Counter(train_bigrams)

    num_unseen_bigram_types = len([bigram for bigram in test_bigram_count if train_bigram_count[bigram] == 0])
    num_unseen_bigram_tokens = sum(test_bigram_count[bigram] for bigram in test_bigram_count if train_bigram_count[bigram] == 0)

    percentage_unseen_bigram_types = (num_unseen_bigram_types / len(test_bigram_count)) * 100
    percentage_unseen_bigram_tokens = (num_unseen_bigram_tokens / len(test_bigrams)) * 100

    return percentage_unseen_bigram_types, percentage_unseen_bigram_tokens

def calculate_unigram_log_prob(sentence, train_count):
    return sum(math.log2(train_count[word] / sum(train_count.values())) for word in sentence)

def calculate_bigram_log_prob(sentence, bigram_train_count, train_count):
    bigram_input_split = list(zip(sentence, sentence[1:]))
    bigram_log_prob = sum(
        math.log2(bigram_train_count.get(bigram, 0) / train_count.get(bigram[0], 0))
        if bigram_train_count.get(bigram, 0) != 0
        else float("-inf")
        for bigram in bigram_input_split
    )
    return bigram_log_prob if bigram_log_prob != float("-inf") else "undefined"

def calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count, vocab_size):
    addone_bigram_log_prob = 0
    addone_bigram_input_split = zip(sentence, sentence[1:])
    for bigram in addone_bigram_input_split:
        addone_bigram_log_prob += math.log2((bigram_train_count[bigram] + 1) / (train_count[bigram[0]] + vocab_size))
    return addone_bigram_log_prob

def calculate_corpus_perplexity(log_probs, num_tokens):
    total_log_prob = sum(log_prob if log_prob != "undefined" else float("-inf") for log_prob in log_probs)
    if total_log_prob == float("-inf"):
        return "undefined"
    return 2 ** (-1 * total_log_prob / num_tokens)

# Read training and test corpora
train_corpus = read_corpus("training_processed.txt")
test_corpus = read_corpus("test_processed.txt")

# Calculate word types and tokens in the training corpus
num_word_types, num_word_tokens = calculate_word_types_and_tokens(train_corpus)
print("Question 1: Number of word types in the training corpus (including </s> and <unk>):", num_word_types)
print("Question 2: Number of word tokens in the training corpus (excluding <s>):", num_word_tokens)

# Calculate percentage of unseen word types and tokens in the test corpus
percentage_unk_types, percentage_unk_tokens = calculate_unseen_percentage(test_corpus, set(train_corpus))
print("Question 3:")
print("Percentage of unseen word types: {:.2f}%".format(percentage_unk_types))
print("Percentage of unseen word tokens: {:.2f}%".format(percentage_unk_tokens))

# Calculate percentage of unseen bigram types and tokens in the test corpus
percentage_unseen_bigram_types, percentage_unseen_bigram_tokens = calculate_unseen_bigrams(test_corpus, set(train_corpus))
print("Question 4:")
print("Percentage of bigram types in the test corpus that did not occur in training: {:.2f}%".format(percentage_unseen_bigram_types))
print("Percentage of bigram tokens in the test corpus that did not occur in training: {:.2f}%".format(percentage_unseen_bigram_tokens))

# Calculate log probabilities for unigram, bigram, and add-one bigram models for a sample sentence
# Question 5
bigram_train_count = Counter(zip(train_corpus, islice(train_corpus, 1, None)))
train_count = Counter(train_corpus)
sentence = "I look forward to hearing your reply ."
sentence = ["<s>"] + sentence.lower().split() + ["</s>"]
sentence = ["<unk>" if word not in train_count else word for word in sentence]
print("Question 5: Log Probability for the sentence under : ")
unigram_log_prob = calculate_unigram_log_prob(sentence, train_count)
print("Unigram model:", unigram_log_prob)

bigram_log_prob = calculate_bigram_log_prob(sentence, bigram_train_count, train_count)
print("Bigram model:", bigram_log_prob)

add_one_bigram_log_prob = calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count, len(train_count))
print("Add-One Bigram model:", add_one_bigram_log_prob)

#Question 6:
def calculate_perplexity(log_prob, sentence):
    if log_prob == "undefined":
        return "undefined"
    return 2 ** (-1 * log_prob / len(sentence))

unigram_prob = calculate_unigram_log_prob(sentence, train_count)
unigram_perplexity = calculate_perplexity(unigram_prob, sentence)
print("Question 6: Perplexity for the sentence under :")
print("Unigram model:", unigram_perplexity)

bigram_prob = calculate_bigram_log_prob(sentence, bigram_train_count, train_count)
bigram_perplexity = calculate_perplexity(bigram_prob, sentence)
print("Bigram model:", bigram_perplexity)

add_one_bigram_prob = calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count, len(train_count))
add_one_bigram_perplexity = calculate_perplexity(add_one_bigram_prob, sentence)
print("Add-One Bigram model:", add_one_bigram_perplexity)

# Calculate perplexity for the entire test corpus under unigram, bigram, and add-one bigram models
print("Question 7:")
test_corpus_num_tokens = sum(len(sentence) for sentence in test_corpus)

test_corpus_unigram_log_probs = [calculate_unigram_log_prob(sentence, train_count) for sentence in test_corpus]
test_corpus_unigram_perplexity = calculate_corpus_perplexity(test_corpus_unigram_log_probs, test_corpus_num_tokens)
print("Perplexity of the entire test corpus under the unigram model:", test_corpus_unigram_perplexity)

test_corpus_bigram_log_probs = [calculate_bigram_log_prob(sentence, bigram_train_count, train_count) for sentence in test_corpus]
test_corpus_bigram_perplexity = calculate_corpus_perplexity(test_corpus_bigram_log_probs, test_corpus_num_tokens)
print("Perplexity of the entire test corpus under the bigram model:", test_corpus_bigram_perplexity)

test_corpus_add_one_bigram_log_probs = [calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count, len(train_count)) for sentence in test_corpus]
test_corpus_add_one_bigram_perplexity = calculate_corpus_perplexity(test_corpus_add_one_bigram_log_probs, test_corpus_num_tokens)
print("Perplexity of the entire test corpus under the add-one bigram model:", test_corpus_add_one_bigram_perplexity)
