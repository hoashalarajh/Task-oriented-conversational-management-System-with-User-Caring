# Import the required functions and classes
from hierarchical_stn import hstn
import random
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# natural language processing block
import spacy
nlp = spacy.load('en_core_web_sm')
import string
punct = string.punctuation
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS) # list of stopwords

# creating a function for data cleaning
def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = [] # list of tokens
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
            tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:    # Stopwords and punctuation removal
            cleaned_tokens.append(token)
    return cleaned_tokens

# sentiment analysis model
import pickle
# Load the saved model using pickle
path = "C://Users/Hoashalarajh/OneDrive - University of Moratuwa\Documents/#ICRA 2023/#Sentiment_model_SVM/"
with open(path + 'emotion_detector.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Define the initial superstate
current_superstate = 0

# Initialize variables to store user responses, emotion, and engagement
user_responses = []
emotion_list = []
engagement_list = []

# Define the number of interactions
num_interactions = 6


# defining a fuction to change the topic based on emotional cue of the user
def topic_changing(sentence):
    prediction = loaded_model.predict(sentence)
    if prediction[0] == "negative":
        return "negative"
    else:
        return "positive"

# couting emotion
def count_emotion(emotion):
    pos = 0
    for i in emotion:
        if i == "positive":
            pos = pos + 1
        else:
            pos = pos + 0
    return pos / 6

def calc_user_engagement(sent):
    total = 0
    new_list = []
    emo_list = []

    for i in range(0, len(sent)):
        current_sent = sent[i]
        result = topic_changing([current_sent])
        emo_list.append(result)
        #print (result)
        cur_sent = current_sent.split()
        total = total + len(cur_sent)
        new_list.append(len(cur_sent))
        #print (len(cur_sent))
        
    #print (total)
    #print (total/6)
    #print (new_list)
    count = 0
    for i in new_list:
        if i >= (total/6):
            count = count + 1
    #print (count)  

def interaction_mode_change(num):
    if num <= 0.32:
        return supporting
    elif num <= 0.74:
        return listening
    else:
        return None
        
# define superstates
#  about lecture
def about_lec():
    sent = hstn.ask_about_lecture()
    fuzzy_output = hstn.calc_fuzzy(count_emotion(sent), calc_user_engagement(sent))
    interaction_mode_change(fuzzy_output)

# Task organization
def task_org():
    sent = hstn.task_organization()
    fuzzy_output = hstn.calc_fuzzy(count_emotion(sent), calc_user_engagement(sent))
    interaction_mode_change(fuzzy_output)

# Student interaction
def student_int():
    sent = hstn.student_interaction()
    fuzzy_output = hstn.calc_fuzzy(count_emotion(sent), calc_user_engagement(sent))
    interaction_mode_change(fuzzy_output)

# clarity
def clarity():
    sent = hstn.clarity()
    fuzzy_output = hstn.calc_fuzzy(count_emotion(sent), calc_user_engagement(sent))
    interaction_mode_change(fuzzy_output)

# listeing to user
def listening():
    sent  = hstn.listening_to_user()

# supporting the user
def supporting():
    sent = hstn.Supporting_user()


## overall finale interative system #############
# starting with greeting
hstn.greeting()
# going wth about lecture
about_lec()
# going with task organization
task_org()
# going with stdent interaction
student_int()
# going with clarity
clarity()
# concluding
hstn.concluding()
