# NLP (Natural Language Processing):
# NLU (Natural Language Understanding): 
    # Ability of understanding human language
# NLG (Natural Language Generation): 
    # Ability to generate human-similar written sentences

# "Hey, what's on the news today?"
# Intent: get_news
# Entity: today

# The machine learning model will be used to recognize 
# the intents and entities of the chat


## Import and loading data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, legacy
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

## Preprocess Data
# Raw data -> data for the machine to easily read

# Method 1: Tokenizing
words, classes, documents = [], [], []
ignore_letters = ['!','?',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents in the corpus
        tag = intent['tag']
        documents.append((word, tag))
        # Add to our classes list
        if tag not in classes:
            classes.append(tag)

print(documents)

# Method 2: Lemmatization
# Convert words into lemma form ('playing', 'plays', 'played' -> all turn into 'play')
# Reduces the total number of words in the vocabulary

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


## Create Training and Testing Data
# Train the model by converting each input pattern -> numbers

# Create the training data
training = []
# Create an empty array for the output
output_empty = [0] * len(classes)
# training set, bag of words for every sentence
for doc in documents:
    # initializing the bag of words
    bag = []
    # list of tokenized words for the pattern
    word_patterns = doc[0]
    # lemmatize each word - create base word, in attempt to represent
    # related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # create the bag of words array with 1, if word is found in current pattern
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# shuffle the features and make numpy array
random.shuffle(training)
training = np.array(training, dtype=object)
# create training and testing lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data is created")

## Training the Model
# Neural network of 3 dense layers:
# 1. 128 neurons
# 2. 64 neurons
# 3. Same amount of neurons as classes

# deep neural networks model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiling model. SGD with Nesterov accelerated gradient gives 
# good results for this model
sgd = legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('model is created')