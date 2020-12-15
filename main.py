import random
import collections
import matplotlib.pyplot
import nltk.classify
from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lem = WordNetLemmatizer() #strips words to their root word

# opening action words file
def action():
    text = open("nltk_data/corpora/movie_genres/action.txt", "r")
    actiontext = text.read()
    text.close()
    # cleaning up the file
    actiontext = actiontext.lower()
    actiontext = lem.lemmatize(actiontext)
    action = word_tokenize(actiontext)
    # removing punctuation
    action = [word for word in action if word.isalpha()]
    # removing frequently used words
    stop_words = set(stopwords.words('english'))
    filteredaction = [a for a in action if not a in stop_words]
    #actfreq = FreqDist(filteredaction)
    #actfreq.plot(10) #frequency distribution for 10 most common words
    return filteredaction

def comedy():
    # opening comedy words file
    text = open("nltk_data/corpora/movie_genres/comedy.txt", "r")
    comedytext = text.read()
    text.close()
    # cleaning up the file
    comedytext = comedytext.lower()
    comedytext = lem.lemmatize(comedytext)
    comedy = word_tokenize(comedytext)
    # removing punctuation
    comedy = [word for word in comedy if word.isalpha()]
    # removing frequently used words
    stop_words = set(stopwords.words('english'))
    filteredcom = [c for c in comedy if not c in stop_words]
    #comfreq = FreqDist(filteredcom)
    #comfreq.plot(10) #frequencydistribution for 10 most common words
    return filteredcom

# opening drama words file
def drama():
    text = open("nltk_data/corpora/movie_genres/drama.txt", "r")
    dramatext = text.read()
    text.close()
    # cleaning up the file
    dramatext = dramatext.lower()
    dramatext = lem.lemmatize(dramatext)
    drama = word_tokenize(dramatext)
    # removing punctuation
    drama = [word for word in drama if word.isalpha()]
    # removing frequently used words
    stop_words = set(stopwords.words('english'))
    filtereddrama = [d for d in drama if not d in stop_words]
    #dramafeatures = FreqDist(filtereddrama)
    #dramafeatures.plot(10) #frequency distribution for 10 most common words
    return filtereddrama

def horror():
    # opening horror words file
    text = open("nltk_data/corpora/movie_genres/horror.txt", "r")
    horrortext = text.read()
    text.close()
    # cleaning up the file
    horrortext = horrortext.lower()
    horrortext = lem.lemmatize(horrortext)
    horror = word_tokenize(horrortext)
    # removing punctuation
    horror = [word for word in horror if word.isalpha()]
    # removing frequently used words
    stop_words = set(stopwords.words('english'))
    filteredh = [h for h in horror if not h in stop_words]
    #horrorfreq = FreqDist(filteredh)
    #horrorfreq.plot(10) #frequency distribution for 10 most common words
    return filteredh

def romance():
    #opening romance words file
    text = open("nltk_data/corpora/movie_genres/romance.txt", "r")
    romancetext = text.read()
    text.close()
    # cleaning up the file
    romancetext = romancetext.lower()
    romancetext = lem.lemmatize(romancetext)
    romance = word_tokenize(romancetext)
    # removing punctuation
    romance = [word for word in romance if word.isalpha()]
    # removing frequently used words
    stop_words = set(stopwords.words('english'))
    filteredr = [r for r in romance if not r in stop_words]
    romfreq = FreqDist(filteredr)
    #romfreq.plot(10) #frequency distribution for 10 most common words
    return filteredr
##############################################################################################

filteredaction = action()
filteredcom = comedy()
filtereddrama = drama()
filteredh = horror()
filteredr = romance()

# create full dataset and randomize
action_set = [(word, "Action") for word in filteredaction]
comedy_set = [(word, "Comedy") for word in filteredcom]
drama_set = [(word, "Drama") for word in filtereddrama]
horror_set = [(word, "Horror") for word in filteredh]
romance_set = [(word, "Romance") for word in filteredr]
keywords = action_set + comedy_set + drama_set + horror_set + romance_set

#shuffle the keywords
random.shuffle(keywords)
#plot 15 most frequent words in the combined set
#keyfreq = FreqDist(keywords)
#keyfreq.plot(15)
#find the words within the dataset
def find_features(keywords):
    f = set(keywords)
    features = {}
    for f in keywords:
        features['Word: ({})'.format(keywords)] = (f in keywords)
    return features

genres = collections.defaultdict(set)
# split training set and testing set
featureset = [(find_features(plot), category) for (plot, category) in keywords]
train_set = featureset[1:1500]
test_set = featureset[2000:4000]

genretest = collections.defaultdict(set)

#NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_set)
print("Training Accuracy: ", nltk.classify.accuracy(classifier, train_set) * 100)
print("Testing Accuracy: ", nltk.classify.accuracy(classifier, test_set) * 100)
classifier.show_most_informative_features(20)
