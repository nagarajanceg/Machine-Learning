import glob
import os
import re
import sys
from collections import Counter

# spam_train_path = "stem/train/spam"
# ham_train_path = "stem/train/ham"
# spam_test_path = "stem/test/spam"
# ham_test_path = "stem/test/ham"

spam_train_path = sys.argv[2]+"/spam"
ham_train_path = sys.argv[2]+"/ham"
spam_test_path = sys.argv[3]+"/spam"
ham_test_path = sys.argv[3]+"/ham"

learning_rate = float(sys.argv[4])
iterations = int(sys.argv[5])
lambda_val = float(sys.argv[6])
spam = 1
ham = 0
# learning_rate = 0.005

# print(learning_rate)
# print(spam_train_path, ham_train_path, spam_test_path, ham_test_path)

# Method to read file content using name
def file_read(filename):
    try:
        # rt - read and text mode
        with open(filename, 'rt', encoding='utf_8') as f:
            return f.read()
    except:
        # to handle certain in binary encoding
        with open(filename, 'rt', encoding='latin_1') as f:
            return f.read()


stop_words = ""
# for filename in glob.glob(os.path.join(sys.argv[1]+"/stop_words", '*.txt')):
stop_words = file_read(sys.argv[1]+"/stop_words.txt").splitlines()
# print(stop_words)
    # stop_words = file_read("stop_words/stop_words.txt").splitlines()


# helper method to remove the stop words from spam/ham class trained words
def filtered_words(class_list_words):
    for word in class_list_words:
        if word in stop_words:
            class_list_words.remove(word)
    return class_list_words


# Method to get all files in a given directory
def parse_file_content(path, stop_words_flag):
    file_content = ""
    map_file_words_count = {}
    for filename in glob.glob(os.path.join(path, '*.txt')):
        content = file_read(filename)
        name = filename.split(path + "/")
        # remove stopwords from file content if the flag is set
        if stop_words_flag:
            # Remove special characters and punctuation
            content = filtered_words(remove_punctuation_tokenizer(content))
        else:
            # Remove special characters and punctuation
            content = remove_punctuation_tokenizer(content)
        content = " ".join(content)
        #map the file with content collection count for each word
        map_file_words_count[name[1]] = Counter(content.split())
        # concat all the file contents in to a single whole content
        file_content += content + " "
    return file_content, map_file_words_count


# Remove special characters
def remove_punctuation_tokenizer(content):
    stem_tokens = []
    for word in content.split():
        word = word.lower()
        stem_tokens += filter(None, re.split(r'\W+|_', word))
    return stem_tokens


def class_classifier(path, stop_words_flag):
    spam_stemmed, content_dict = parse_file_content(path, stop_words_flag)
    spam_stemmed = list(spam_stemmed.split())
    # spam_stemmed = do_stemming(remove_punctuation_tokenizer(content_to_stem))
    return spam_stemmed, content_dict


lt = {}
lw = {}
# weight holder for each word
lw_weight = {}
predict_value = {}


def init_weights(elements):
    for i in elements:
        lw[i] = 0
        lw_weight[i] = 0
        lt[i] = 0
    return lt


def find_predict_value(class_list):
    for key in class_list.keys():
        sum = 1
        for ckey, cval in class_list[key].items():
            sum = sum + cval * lt.get(ckey)
        if sum > 0:
            predict_value[key] = 1
        else:
            predict_value[key] = 0


def classify(list):
    sum = 0
    for key, value in list.items():
        if key in lt:
            sum += value * lt.get(key)
    if sum > 0:
        return 1
    else:
        return 0


def dw_calculate(spam_list, ham_list, unique_list):
    dw = {}
    for i in unique_list:
        dw[i] = 0
    for i in dw.keys():
        for skey, sval in spam_list.items():
            frequency = sval[i]
            dw[i] = dw[i] + frequency * (spam - predict_value[skey])
        for skey, sval in ham_list.items():
            frequency = sval[i]
            dw[i] = dw[i] + frequency * (ham - predict_value[skey])
    return dw


def update_weights(dw):
    for key, value in lt.items():
        modified_val = value + float(learning_rate) * (float(dw[key]) - (float(lambda_val) * float(value)))
        lt[key] = modified_val


def compute_helper(stop_flag):
    spam_classify_list, spam_dict = class_classifier(spam_train_path, stop_flag)
    # print(spam_dict)
    ham_classify_list, ham_dict = class_classifier(ham_train_path, stop_flag)
    # print(len(spam_dict))
    # print(len(ham_dict))
    # print(ham_dict)
    combined_list = spam_classify_list + ham_classify_list
    unique_list = sorted(set(combined_list))
    init_weights(unique_list)
    return spam_dict, ham_dict, unique_list


def compute_lr_with_out_stop_words():
    reset_globals()
    spam_dict, ham_dict, unique_list = compute_helper(False)
    compute_lr_iterations(spam_dict, ham_dict, unique_list)


def compute_lr_with_stop_words():
    reset_globals()
    spam_dict, ham_dict, unique_list = compute_helper(True)
    compute_lr_iterations(spam_dict, ham_dict, unique_list)


def compute_lr_iterations(spam_dict, ham_dict, unique_list):
    for i in range(iterations):
        find_predict_value(spam_dict)
        find_predict_value(ham_dict)
        dweight = dw_calculate(spam_dict, ham_dict, unique_list)
        update_weights(dweight)


def apply_lr_helper(class_dict, class_val):
    count = 0
    for key, value in class_dict.items():
        predict = classify(value)
        if predict == class_val:
            count += 1
    return count


def reset_globals():
    lt.clear()
    lw.clear()
    lw_weight.clear()


def apply_lr(stop_flag):
    predicted_count = 0
    print("executing..")
    spam_test_content, spam_test_dict = class_classifier(spam_test_path, stop_flag)
    ham_test_content, ham_test_dict = class_classifier(ham_test_path, stop_flag)

    predicted_count += apply_lr_helper(spam_test_dict, spam)
    predicted_count += apply_lr_helper(ham_test_dict, ham)
    total = len(spam_test_dict) + len(ham_test_dict)
    accuracy = float(predicted_count) / float(total) * 100
    print("Accuracy ", accuracy)


def evaluate_lr_with_out_stop_words():
    apply_lr(False)


def evaluate_lr_with_stop_words():
    apply_lr(True)

def predict_weights(counter_list):
    sum = 0
    for key, value in counter_list.items():
        if key in lw_weight:
            sum += lw_weight[key] * value
    if sum > 0:
        return 1
    else:
        return 0

def update_percp_weight(predicted, actual, ct_list ):
    error = actual - predicted
    for key, value in ct_list.items():
        if key in lw_weight:
            lw_weight[key] += float(learning_rate) * error * value

def learning_weight(class_dict, class_val):
    # print("learning")
    for key, value in class_dict.items():
        prediction = predict_weights(value)
        update_percp_weight(prediction, class_val, value)


def compute_percp_iterations(spam_dict, ham_dict, unique_list):
    for i in range(iterations):
        learning_weight(spam_dict, spam)
        learning_weight(ham_dict, ham)
        # print(lw_weight)

def apply_percp_helper(class_dict, class_val):
    count = 0
    for key, value in class_dict.items():
        predict = predict_weights(value)
        if predict == class_val:
            count += 1
    return count


def apply_perceptron(stop_flag):
    predicted_count = 0
    print("executing..")
    spam_test_content, spam_test_dict = class_classifier(spam_test_path, stop_flag)
    ham_test_content, ham_test_dict = class_classifier(ham_test_path, stop_flag)

    predicted_count += apply_percp_helper(spam_test_dict, spam)
    predicted_count += apply_percp_helper(ham_test_dict, ham)
    total = len(spam_test_dict) + len(ham_test_dict)
    accuracy = float(predicted_count) / float(total) * 100
    print("Accuracy ", accuracy)

def compute_percp_no_stop_words():
    reset_globals()
    spam_dict, ham_dict, unique_list = compute_helper(False)
    compute_percp_iterations(spam_dict, ham_dict, unique_list)

def compute_percp_stop_words():
    reset_globals()
    spam_dict, ham_dict, unique_list = compute_helper(True)
    compute_percp_iterations(spam_dict, ham_dict, unique_list)

def evaluate_percp_no_stop_words():
    apply_perceptron(False)

def evaluate_percp_stop_words():
    apply_perceptron(True)


if __name__ == "__main__":
    print("LR No Stop Words", end=" ")
    compute_lr_with_out_stop_words()
    evaluate_lr_with_out_stop_words()

    print("LR With Stop Words", end=" ")
    compute_lr_with_stop_words()
    evaluate_lr_with_stop_words()

    print("Perceptron No Stop Words", end=" ")
    compute_percp_no_stop_words()
    evaluate_percp_no_stop_words()
    print("Perceptron With Stop Words", end=" ")
    compute_percp_stop_words()
    evaluate_percp_stop_words()
