# CSCI366/CSCI780 Homework 1 README
### Part 1
# Calculating Bigram Probability with Add-One Smoothing

## Introduction
This Python script calculates the probability of a specific bigram ("Sam am") in a given corpus using add-one smoothing. Link for the question is exercise 3.4 from Chapter 3 in the textbook:
"https://web.stanford.edu/~jurafsky/slp3/3.pdf"

## Instructions
1. Ensure you have Python 3.1 or later installed on your system.
2. Download the `part_1.py` file and place it in a directory.
3. Open a terminal or command prompt and navigate to the directory containing the Python script.

## Running the Code
To run the script and calculate the probability of the bigram "Sam am" using add-one smoothing, follow these steps:

1. Open the `part_1.py` file in a text editor.
2. Run the script using the following command:
```
    python part_1.py
```

### Part 2:
## 1.1 Preprocessing

### Python Code Instructions

1. Ensure you have Python 3.1 or later installed on your system.
2. Download the `pre_processing.py` file and place it in a directory along with your training and test corpus files (`train-Spring2024.txt` and `test.txt`).
3. Open a terminal or command prompt and navigate to the directory where `pre_processing.py` and your corpus files are located.

### Running the Code
To run the preprocessing script and obtain the processed corpora, follow these steps:

1. Open the `pre_processing.py` file in a text editor.
2. Modify the file path in the `read_corpus` function to point to your corpus file.
3. Save the changes and close the text editor.
4. In the terminal or command prompt, run the `pre_processing.py` file using the following command:
```
    python pre_processing.py
```

### Output
The script will process the training and test corpora by adding start and end symbols to each sentence, lowercasing all words, replacing words occurring once in the training corpus with `<unk>`, and replacing unseen words in the test corpus with `<unk>`. The processed corpora will be saved in `training_processed.txt` and `test_processed.txt`, respectively.

## 1.2 TRAINING THE MODELS

### Introduction
This Python script trains three different language models (unigram maximum likelihood, bigram maximum likelihood, and bigram with Add-One smoothing) on a given corpus.

### Instructions
1. Ensure you have Python installed on your system (version 3.1 or later).
2. Download the `train-Spring2024.txt` file and place it in the same directory as the Python script (`traning_model.py`).
3. Open a terminal or command prompt and navigate to the directory containing the Python script and the corpus file.

### Running the Code
To run the script and train the language models, use the following command 
```
    python traning_model.py
```

### Output
The script will calculate the probabilities for the unigram, bigram, and Add-One smoothed bigram models for a few example phrases. The output will display the probabilities for each model.

### Dependencies
- Python 3.1 or later
- `collections.Counter` for counting occurrences of tokens
- `math` for mathematical operations

## 1.3 QUESTIONS

### Introduction
This Python script evaluates different language models (unigram, bigram, and bigram with Add-One smoothing) on a given corpus and test set. It calculates various metrics such as word types and tokens, percentage of unseen words, and perplexity for each model.

### Instructions
1. Ensure you have Python 3.1 or later installed on your system.
2. Download the `training_processed.txt` and `test_processed.txt` files and place them in the same directory as the Python script (`questions.py`).
3. Open a terminal or command prompt and navigate to the directory containing the Python script and the corpus files.

### Running the Code
To run the script and evaluate the language models, use the following command:
```
    python questions.py
```

### Output
The script will calculate and display all the answers for the given questions.

### Dependencies
- Python 3.1 or later
- `collections.Counter` for counting occurrences of tokens
- `itertools.islice` for iterating over the corpus in sliding windows
- `math` for mathematical operations

