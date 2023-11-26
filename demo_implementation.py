import time
import warnings

###############################
from simpful import *

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




##################################

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


# defining a fuction to change the topic based on emotional cue of the user
def topic_changing(sentence):
    prediction = loaded_model.predict(sentence)
    if prediction[0] == "negative":
        return "negative"
    else:
        return "positive"
    
def sent_length(sentence):
   sentence = sentence.split()
   #print (sentence)
   #print (len(sentence))
   return (len(sentence))

sent_list = []
    

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
    
    print (f"Recent emotion is {count_emotion(emo_list)}")
    print (f"user engagement is : {count /6}")

# About lecture superstate
print ("\n")
inp = input("User : ")
time.sleep(1)

print ("System A : Hi, Good Day today...")
inpt = input("User : ")
time.sleep(1)



print ("Current Superstate: About lecture")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : Happy to have you here, We can start our conversation")
inp = input("User : ")
time.sleep(1)
sent_list.append(inp)

print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")

print ("Current Superstate: About lecture")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : Can you share me your name, department, which year and semester are you in ?")
inp = input ("User : ")
time.sleep(1)
sent_list.append(inp)

print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")

print ("Current Superstate: About lecture")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : Can you share briefly about the lecture you had today ?")
inp = input ("User : ")
time.sleep(1)
sent_list.append(inp)

print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")

print ("Current Superstate: About lecture")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : What was the interesting thing you learned in the lecture ?")
inp = input ("User : ")
time.sleep(1)
sent_list.append(inp)

print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")

print ("Current Superstate: About lecture")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : Was there anything in the lecture that you found difficult to understand?")
inp = input ("User : ")
time.sleep(1)
sent_list.append(inp)

print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")

print ("Current Superstate: About lecture")
print ("State swticher: Active")
time.sleep(1)
print ("System A : Did you enjoy the lecture today ?")
inp = input ("User : ")
time.sleep(1)
sent_list.append(inp)

print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")

calc_user_engagement(sent_list)
# assigning values
# val1, val2 = emotion, engagement
val1, val2 = 0.5, 0.3333
# compute interaction mode
mode = calc_fuzzy(val1, val2)
mode_numeric = mode["Tip"]
# printing the result
print (f"For RecentEmotion = {val1} and UserEngagement = {val2} the Interaction mode is {mode_numeric}")
print ("Next Superstate: Supporting User")
print ("\n")


#supporting user superstate
print ("Current Superstate: Supporting User")
print ("State swticher: Not active")
time.sleep(1)
print ("I understand that the concepts are really difficult to understand in the lecture.")
time.sleep(1)
print ("Anyway don't miss any of the lectures. At least you will learn something in a lecture rather than avoiding it")
time.sleep(1)
print ("However, if you have suggestions about the lectures feel free to speak about them...")
time.sleep(1)
inp = input ("User : ")
time.sleep(1)


print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")


# back into About lecture superstate
print ("Current Superstate: About lecture")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : Nice to hear, Do you think the lecture is useful and fun to learn ?")
inp = input ("User : ")
time.sleep(1)


print (f"Current Sentiment: {topic_changing([inp])}")
print (f"Length of sentence: {sent_length(inp)}")
print ("\n")

# superstate About lecture Ends with this

# Moving to next superstate Tak organization
print ("About lecture superstate ended")
time.sleep(1)
print ("Current Superstate: Task organization")
print ("State swticher: Not active")
time.sleep(1)
print ("System A : Okay we will move to the next topic, can you share with me the speed of the subject matter is sufficient enough ?")

inp = input ("User : ")
print ("\n")
