import glob
import os
import math
import sys
import re
# from nltk.stem.porter import *

# spam_train_path = "stem/train/spam"
# ham_train_path = "stem/train/ham"
# spam_test_path = "stem/test/spam"
# ham_test_path = "stem/test/ham"

spam_train_path = sys.argv[2]+"/spam"
ham_train_path = sys.argv[2]+"/ham"
spam_test_path = sys.argv[3]+"/spam"
ham_test_path = sys.argv[3]+"/ham"

# Method to read file content using name
def file_read(filename):
    try:
        # rt - read and text mode
        with open(filename, 'rt', encoding='utf_8') as f:
            # print("fileName == ", filename)
            return f.read()
    except:
        # to handle certain in binary encoding
        with open(filename, 'rt', encoding='latin_1') as f:
            return f.read()

# stop_words = file_read("stop_words.txt").splitlines()
stop_words = ""
for filename in glob.glob(os.path.join(sys.argv[1], '*.txt')):
        stop_words = file_read(filename).splitlines()

# Method to get all files in a given directory
def parse_file_content(path):
    file_content = ""
    for filename in glob.glob(os.path.join(path, '*.txt')):
        file_content += file_read(filename) + " "
        ## concat all the file contents in to a single whole content
    return file_content


# Get the total number of files available in a given path
def get_file_count(path):
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return num_files

# Remove special characters
def remove_punctuation_tokenizer(content):
    stem_tokens = []
    for word in content.split():
        word = word.lower()
        stem_tokens += filter(None, re.split(r'\W+|_|-', word))
    return stem_tokens


def spam_classifier():
    content_to_stem = parse_file_content(spam_train_path)
    content_to_stem = content_to_stem.lower()
    spam_stemmed = remove_punctuation_tokenizer(content_to_stem)
    # spam_stemmed = do_stemming(remove_punctuation_tokenizer(content_to_stem))
    return spam_stemmed


def ham_classifier():
    content_to_stem = parse_file_content(ham_train_path).lower()
    # ham_stemmed = do_stemming(remove_punctuation_tokenizer(content_to_stem))
    ham_stemmed = remove_punctuation_tokenizer(content_to_stem)
    return ham_stemmed

# calculating prior for both spam and ham
def calculate_prior(spath, hpath):
    spam_doc_count = get_file_count(spath)
    ham_doc_count = get_file_count(hpath)
    total_doc_count = spam_doc_count + ham_doc_count
    prob_spam = spam_doc_count / total_doc_count
    prob_ham = ham_doc_count / total_doc_count
    prior = {}
    prior["spam"] = prob_spam
    prior["ham"] = prob_ham
    return prior

# calculate occurrence frequency
def count_tokens(word, dataset):
   return dataset.count(word.lower())


# helper method to calculate conditional probability for every element belong to the spam/ham class
def individual_cond_prob(class_frequency, class_list, overall_length):
    prob = {}
    for label, value in class_frequency.items():
        prob[label] = (value + 1) / (len(class_list) + overall_length)
    return prob

# figure out the each word frequency and their conditional probability
def compute_class_attributes(overall_word_list, overall_length, class_list):
    freq = {}
    for i in overall_word_list:
        freq[i] = count_tokens(i, class_list)
    prob = individual_cond_prob(freq, class_list, overall_length)
    return prob

#classify the given test data to match with trained data
def classify(data, prior, prob_class_data):
    occurrence = math.log(prior, 2)
    for word in data:
        cp_word = prob_class_data.get(word)
        if cp_word is not None:
            occurrence += math.log(cp_word, 2)
    return occurrence

#helper method used to classify the test data with both class
def determine_class(data, prior, prob_spam, prob_ham):
    spam = classify(data, prior.get('spam'), prob_spam)
    ham = classify(data, prior.get('ham'), prob_ham)
    #Return based on max presence class count
    if spam > ham:
        return 'spam'
    else:
        return 'ham'


def apply_multinomial(prior, prob_spam, prob_ham):
    #get a list of filenames from a directory
    files = os.listdir(spam_test_path)
    hfiles = os.listdir(ham_test_path)
    accuracy = 0
    for name in files:
        # spam_test_data = do_stemming(remove_punctuation_tokenizer(file_read(os.path.join(spam_test_path, name))))
        spam_test_data = remove_punctuation_tokenizer(file_read(os.path.join(spam_test_path, name)))
        result = determine_class(spam_test_data, prior, prob_spam, prob_ham)
        if result == "spam":
            accuracy += 1

    for name in hfiles:
        # ham_test_data = do_stemming(remove_punctuation_tokenizer(file_read(os.path.join(ham_test_path, name))))
        ham_test_data = remove_punctuation_tokenizer(file_read(os.path.join(ham_test_path, name)))
        result = determine_class(ham_test_data, prior, prob_spam, prob_ham)
        if result == "ham":
            accuracy += 1
    return accuracy

#helper method to remove the stop words from spam/ham class trained words
def filtered_words(stop_words, class_list_words):
    for word in class_list_words:
        if word in stop_words:
            class_list_words.remove(word)
    return class_list_words

def compute_with_stop_words(prior, distinct_words, spam_classify_list,ham_classify_list, word_length ):
    filtered_spam_list = spam_classify_list[:]
    filtered_ham_list = ham_classify_list[:]
    spam_list = filtered_words(stop_words, filtered_spam_list)
    ham_list = filtered_words(stop_words, filtered_ham_list)
    spam_list_prob = compute_class_attributes(distinct_words, word_length, spam_list)
    ham_list_prob = compute_class_attributes(distinct_words, word_length, ham_list)
    evaluate_naive(prior, spam_list_prob, ham_list_prob)

def evaluate_naive(prior, spam_prob, ham_prob):
    acc = apply_multinomial(prior, spam_prob, ham_prob)
    total_test_file_count = get_file_count(spam_test_path) + get_file_count(ham_test_path)
    final_accuracy = float(acc) / float(total_test_file_count)
    print("final accuracy == ", final_accuracy)

spam_classify_list = spam_classifier()
ham_classify_list = ham_classifier()

distinct_words = sorted(set(spam_classify_list + ham_classify_list))
word_length = len(distinct_words)
print("training with data")
spam_prob = compute_class_attributes(distinct_words, word_length, spam_classify_list)
ham_prob = compute_class_attributes(distinct_words, word_length, ham_classify_list)
prior = calculate_prior(spam_train_path, ham_train_path)
print("Evaluating with out stop words")
evaluate_naive(prior, spam_prob, ham_prob)
print("Evaluating with stop words")
compute_with_stop_words(prior, distinct_words, spam_classify_list, ham_classify_list, word_length )

"""
## The following code is used to stem and store in different folder with same file name.
## Later the files can moved to the corresponding train and test folder 
    ##Stemming code:
spam_train_new_path = "stem/"
ham_train_new_path = "stem/"
spam_test_new_path = "stem/"
ham_test_new_path = "stem/

def file_parser(path, target_path):
    for filename in glob.glob(os.path.join(path, '*.txt')):
        file_content = do_stemming(file_read(filename))
        write_data(file_content, target_path+filename)
        
def do_stemming(content):
    stemmer = PorterStemmer()
    stemmer_content = [stemmer.stem(element) for element in content.split()]
    return stemmer_content
    
def write_data(list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file = open(path, "w", encoding="utf-8")
    for item in list:
        file.write("%s\n"%item)
        
file_parser(spam_train_path, spam_train_new_path)
file_parser(ham_train_path, ham_train_new_path)
file_parser(spam_test_path, spam_test_new_path)
file_parser(ham_test_path, ham_train_new_path)
"""
