# importing dependencis
from simpful import *
import random as rn
import os
import pyttsx3
import spacy
import time
import warnings

# importing non-heavy libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# filter warnings
warnings.filterwarnings('ignore')

# loading some corpus
nlp = spacy.load('en_core_web_sm')

# data cleaning
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

# tokenizer=text_data_cleaning, tokenization will be done according to this function
tfidf = TfidfVectorizer(tokenizer = text_data_cleaning, token_pattern = None)

# load the saved model from file using pickle
with open("C://Users/Hoashalarajh/OneDrive - University of Moratuwa/Documents/#ICRA 2023/#Sentiment_model_SVM/emotion_detector.pkl", 'rb') as f:
    loaded_model = pickle.load(f)

# Greeting stage
def greeting():
    greetings = ["Hello", "Hi", "Hi there", "Hello there", "Hey", "Good day", "It's a pleasure to meet you", "Nice to see you", "Howdy"]
    random_response = rn.randint(0, len(greetings) - 1)
    print("System: " + greetings[random_response])
    return (greetings[random_response])

# Feedforward Stage
def feedforward():
    feedforwards = ["How do you do ?", "How are you ?", "How it's going ?"]
    random_feedforward = rn.randint(0, len(feedforwards) - 1)
    return (feedforwards[random_feedforward])

def about_MIRob():
    text = "I am Moratuwa Intelligent Robot, shortly known as MIRob. I am a joint product of the many research students of Intelligent Service Robotics group of University of Moratuwa. I will be asking some questions about the lecture experience you had recently"
    print("System: " + text)
    return (text)

def nice_hear():
    return ("It is nice to hear")   

def text_to_speech(text):
    # Convert the text to speech
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    # say anything - talk something
    print ("Prompt Something .....\n")
    transcript = input("prompt here: ")
    return (transcript) 

def about_me():
     text_to_speech(about_MIRob)


def calc_fuzzy(val1, val2):
    # A simple fuzzy inference system for the tipping problem
    # A simple fuzzy inference system for the tipping problem
    # Create a fuzzy system object
    FS = simpful.FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = simpful.FuzzySet(points=[[0., 1.], [0.19, 1.], [0.49, 0.0]], term='Negative')
    S_2 = simpful.FuzzySet(points=[[0.203 , 0.], [0.503 , 1.], [0.8023, 0.]], term="Neutral")
    S_3 = simpful.FuzzySet(points=[[0.5011, 0.], [0.798 , 1.], [1., 1.]], term="Positive")
    FS.add_linguistic_variable("Emotion", simpful.LinguisticVariable([S_1, S_2, S_3], concept="Recent Emotion", universe_of_discourse=[0,1]))

    F_1 = simpful.FuzzySet(points=[[0., 1.], [0.097, 1.0], [0.2974, 0]], term="Very_low")
    F_2 = simpful.FuzzySet(points=[[0.157, 0.0], [0.34, 1.0], [0.5043, 0.0]], term="Low")
    F_3 = simpful.FuzzySet(points=[[0.4, 0.0], [0.5, 1], [0.6008, 0.0]], term="Medium")
    F_4 = simpful.FuzzySet(points=[[0.5, 0.0], [0.758, 1.0], [0.851, 0.0]], term="High")
    F_5 = simpful.FuzzySet(points=[[0.702, 0.0], [0.9009, 1], [1.0, 1.0]], term="Very_High")
    FS.add_linguistic_variable("Engagement", simpful.LinguisticVariable([F_1, F_2, F_3, F_4, F_5], concept="User Engagement", universe_of_discourse=[0,1]))

    # Define output crisp values
    FS.set_crisp_output_value("Supporting", 0.1)
    FS.set_crisp_output_value("Listening", 0.5)
    FS.set_crisp_output_value("Main", 1.0)

    # Define function for generous tip (food score + service score + 5%)
    #FS.set_output_function("generous", "Food+Service+5")

    # Define fuzzy rules
    R1 = "IF (Emotion IS Negative) AND (Engagement IS Very_low) THEN (Tip IS Supporting)"
    R2 = "IF (Emotion IS Negative) AND (Engagement IS Low) THEN (Tip IS Supporting)"
    R3 = "IF (Emotion IS Negative) AND (Engagement IS Medium) THEN (Tip IS Supporting)"
    R4 = "IF (Emotion IS Negative) AND (Engagement IS High) THEN (Tip IS Listening)"
    R5 = "IF (Emotion IS Negative) AND (Engagement IS Very_High) THEN (Tip IS Listening)"
    R6 = "IF (Emotion IS Neutral) AND (Engagement IS Very_low) THEN (Tip IS Supporting)"
    R7 = "IF (Emotion IS Neutral) AND (Engagement IS Low) THEN (Tip IS Supporting)"
    R8 = "IF (Emotion IS Neutral) AND (Engagement IS Medium) THEN (Tip IS Listening)"
    R9 = "IF (Emotion IS Neutral) AND (Engagement IS High) THEN (Tip IS Main)"
    R10 = "IF (Emotion IS Neutral) AND (Engagement IS Very_High) THEN (Tip IS Main)"
    R11 = "IF (Emotion IS Positive) AND (Engagement IS Very_low) THEN (Tip IS Supporting)"
    R12 = "IF (Emotion IS Positive) AND (Engagement IS Low) THEN (Tip IS Supporting)"
    R13 = "IF (Emotion IS Positive) AND (Engagement IS Medium) THEN (Tip IS Listening)"
    R14 = "IF (Emotion IS Positive) AND (Engagement IS High) THEN (Tip IS Main)"
    R15 = "IF (Emotion IS Positive) AND (Engagement IS Very_High) THEN (Tip IS Main)"
    FS.add_rules([R1, R2, R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15])


    # Set antecedents values
    FS.set_variable("Emotion", val1)
    FS.set_variable("Engagement", val2)

    # Perform Sugeno inference and print output
    return(FS.Sugeno_inference(["Tip"]))

def user_opinion(user_answers):
    positive = 0
    for i in user_answers:
        prediction = loaded_model.predict([i])
        #print (prediction)
        if prediction == "positive":
            positive = positive + 1
    return positive / 6


def calc_user_engagement(user_answers):
        response_len = 0
        for i in user_answers:
            response_len = (response_len + len(i.split(" ")))
        avg_length = response_len / 6

        count = 0
        for i in user_answers:
            i = len(i.split(" "))

            if i >= avg_length:
                count = count + 1
        return count / 6  

# user responses declaration
user_answers = ["Happy to talk with you",
                    "I am Madhesh, from Department of Electrical Engineering. I am now in my final year of my undergraduate",
                    "Yeah, we had control systems lecture today, It is taught by professor veeraragavan",
                    "Actually, Nothing interesting",
                    "Yeah, the concepts are really difficult to understand",
                    "It is boring to me"]

# user response version#2
# user responses declaration
user_answers2 = ["Yes, I have a suggestion. I believe incorporating animated visualizations to help illustrate the concepts would be highly beneficial and enhance the learning experience"]



# one of the main task
def ask_about_lecture():
    about_lec_responses = ["Happy to have you here, we can start our conversation",
                            "Can you share me your name, department, which year and semester are you in ?",
                            "Can you share briefly about the lecture you had today ?",
                            "What was the interesting thing you learned in the lecture ?",
                            "Was there anything in the lecture that you found difficult to understand?",
                            "Did you enjoy the lecture today ?"]
    
    
    for i in range(len(about_lec_responses)):
        time.sleep(2)
        print ("System: " + about_lec_responses[i])
        print("\n")
        text_to_speech(about_lec_responses[i])
        time.sleep(2)
        print ("User: " + user_answers[i])
        print("\n")
        #text_to_speech(user_answers[i])


# one of the alternate task
def supporting_user():
    about_lec_responses = ["I understand that concepts were really difficult to understand in the lecture. Anyway don't miss any of the lectures. At least you will learn something in a lecture rather than avoiding it. However, of you have any suggestions about the lectures feel free to speak about them with me"]
    
    
    for i in range(len(about_lec_responses)):
        time.sleep(10)
        print ("\nSystem: " + about_lec_responses[i])
        print("\n")
        text_to_speech(about_lec_responses[i])

    time.sleep(2)    
    print ("User: " + user_answers2[0])
    print("\n")


# interaction mode decide
def interaction_mode_change(num):
    if num <= 0.32:
        return "supporting"
    elif num <= 0.74:
        return "listening"
    else:
        return None
        


# MIRob starts to chat
#####################################
text_to_speech(greeting())
text_to_speech(feedforward())
query = speech_to_text()
text_to_speech(nice_hear())
text_to_speech(about_MIRob())
# main task
ask_about_lecture()
# decision making
user_op = user_opinion(user_answers)
user_eng = calc_user_engagement(user_answers)
fuzzy_value = calc_fuzzy(user_op, user_eng)
print ("###################################################")
print ("The fuzzy value is: ", fuzzy_value["Tip"])
print ("###################################################")

# checking
interaction_mode = interaction_mode_change(fuzzy_value["Tip"])
if interaction_mode == "supporting":
    print ("##########################################################################################################################")
    print ("The interaction mode is 'Supporting User', so now the system will support the user to express about his opinions")
    print ("##########################################################################################################################")
    supporting_user()

time.sleep(2)
print ("System: Thank you for your suggestion, by the way, Do you think the lecture is useful and fun to learn ?")
text_to_speech("Thank you for your suggestion, by the way, Do you think the lecture is useful and fun to learn ?")   


