# Geneeric imports
import json
import pickle
import random
import numpy as np

# NLP imports
import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('popular', quiet=True) # for downloading popular packages

# Download packages (only the first time)
# nltk.download('punkt') 
# nltk.download('wordnet') 

# NN imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from keras.models import load_model


class Chatbot:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

        # Information
        self.data = None
        try:
            self.model = load_model('chatbot.h5')
            self.data = json.loads(open("data.json").read())
            self.words = np.array(sorted(pickle.load(open('words.pkl','rb'))))
            self.cls = np.array(sorted(pickle.load(open('cls.pkl','rb'))))
        except OSError:
            self.model, self.data, self.words, self.cls = None, None, None, None
        self.docs = None
        
    def train(self):
        with open("data.json", "r") as file:
            self.data = json.loads(file.read())
        self.words, self.cls, self.docs = self.__pre_process()
        training_data = self.__training_data()
        self.model, hist = self.__create_nn_model(*training_data)
        # Save information
        pickle.dump(self.words, open('words.pkl','wb')) 
        pickle.dump(self.cls, open('cls.pkl','wb'))
        self.model.save('chatbot.h5', hist) 

    def __pre_process(self):
        words, cls, docs, ignore = set(), set(), [], ("?", "!")
        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                
                # Tokenize words
                t = nltk.word_tokenize(pattern) 
                filtered = map(lambda w: self.lemmatizer.lemmatize(w.lower()), list(filter(lambda x: x.lower() not in ignore, t)))
                # add to the word list
                words.update(filtered)
                # add to the docs
                docs.append((t, intent["tag"]))
                # add to the classes
                cls.add(intent["tag"])
    
        # Sort elements
        words, cls = np.array(sorted(list(words))), np.array(sorted(list(cls)))
    
        print(len(words), len(cls), len(docs))
        return words, cls, docs

    def __training_data(self):
        training = []
        for doc in self.docs:
            lem_words = list(map(lambda w: self.lemmatizer.lemmatize(w.lower()), doc[0]))
            # Creaate bag of words
            bow = [1 if w in lem_words else 0 for w in self.words]
            # Class which  the document belongs
            output = (self.cls==doc[1]).astype(int).tolist()
            training.append([bow, output])
        
        # shuffle the training data
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Return X, y
        return training[:,0].tolist(), training[:,1].tolist()
    
    def __create_nn_model(self, training_x, training_y) -> Sequential:
        """Create a NN model based on the training data 
        Sequential NN with one input layer, one hidden layer, an output layer and two dropout layers

        Args:
            training_x (np.array): traning matrix X
            training_y (np.array): training vector y

        Returns:
            Sequential: NN model
        """
        model = Sequential()
        # Input layer (size is the amount of docs and shape is the length of the bow)
        model.add(Dense(len(self.docs), input_shape=(len(training_x[0]), ), activation="relu"))
        # Dropout layer (excludes nodes to prevent overfitting)
        model.add(Dropout(.5))
        # Hidden dense layer (all nodes receives from each node an input)
        model.add(Dense(64, activation="relu"))
        # Dropout layer
        model.add(Dropout(.5))
        # Output dense layer (size is the length of y)
        model.add(Dense(len(training_y[0]), activation="softmax"))
        
        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        #fitting and saving the model 
        return model, model.fit(x=np.array(training_x), y=np.array(training_y), epochs=200, batch_size=5, verbose=1)
        
    def run(self):
        print("Este es un bot!")
        while True:
            i = str(input("> ").lower().strip())
            if not i:
                print("Re enter sentence")
            elif i == "exit":
                break
            else:
                bow = self.__process_sentence(i)
                cls = self.__predict_class(bow)
                print(f"Bot: {cls}"+'\n')
                
    def __process_sentence(self, sentence):
        # tokenize given sentence and lemmatize
        lem_words = list(map(lambda w: self.lemmatizer.lemmatize(w.lower()), nltk.word_tokenize(sentence)))
        # calculate bow of the sentence
        return np.array([1 if w in lem_words else 0 for w in self.words])
    
    def __predict_class(self, bow):
        result = self.model.predict(np.array([bow]))
        print(result)
        error = 0.25
        results = [[self.cls[i],r] for i,r in enumerate(result[0]) if r>error]
        print(results)
        
        return results[0][0]
    
        
            


if __name__ == "__main__":
    # Create WordNetLemmatizer
    c = Chatbot()
    # c.train()
    c.run()

