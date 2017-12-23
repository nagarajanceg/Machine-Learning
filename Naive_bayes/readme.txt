Please remember to include the following at the beginning of your type-written document and e-SIGN it by typing your name. Homeworks that do not include this will not be graded!
I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else.
I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record.
I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense.

Gopal Nagarajan


To run
python3 naive.py stop_words train test

Approach:
The train and test spam , ham data stemmed with porter stemmer library from nltk package.
In stemming process each and every file is processed and stored in the same name in another directory.
Then again the stemmed file content read in traning and undergoing the removal of special characters available.
Prior for the data set determined. After that the occurrences of each word is counted and conditional probability is calculated.
once these process done in training and it can apply to the test data set to determine the spam or ham class for every file.

Now the second type of filtering the stop words from the training data set and then ro do the counting, conditional
probability. This trained model can used to evaluate the test data available.
