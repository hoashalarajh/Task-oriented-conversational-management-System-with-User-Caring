import random as rn
from simpful import *
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



# defining a list to store the responses given by user 
sent = []

class hstn():

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

    # Main state - Greeting
    def greeting():
        greeting_responses = ["Hi!","Good morning","Hey, what's up?","Greetings!","How are you doing?","Hi, there! How's your day going?",
                              "Hey, nice to see you!","Well, hello to you too!","Hi, how can I assist you today?","Hello, friend!",
                              "How's everything?","Hey, it's great to see you again!","Hi there, what's new?","How's your day been so far?",
                              "Hello, how's life treating you?"]
        print (greeting_responses[rn.randint(0,len(greeting_responses)-1)])
    
    # superstate - Ask about lecture
    def ask_about_lecture(sent):
        about_lec_responses = ["What were the main key points or takeaways from today's lecture?","Can you summarize the lecture in your own words?",
                               "Were there any concepts or ideas discussed in the lecture that you found challenging or unclear? If so, please explain.",
                               "How do you think the material covered in this lecture relates to previous lectures or topics we've discussed in the course?",
                               "Can you provide an example or real-life application of one of the concepts discussed in the lecture?",
                               "Did the lecturer mention any open questions or areas of research related to the topic? What are your thoughts on those?",
                               "How would you apply what you've learned in today's lecture to solve a specific problem or scenario?",
                               "Were there any interesting or surprising insights from the lecture that you'd like to share or discuss further?",
                               "Can you identify any connections between the content of this lecture and current events or trends in the field?",
                               "Do you have any questions or concerns about the material that was covered today?"]
        for i in about_lec_responses:
            sent.append(sent)
            print (i)
            sent = input()
            sent.appned(sent)

    # superstate - Ask about task organization
    def task_organization():
        # superstate - Task organization
        task_org_responses = ["Speed of the sessions/time management were reasonable","Relevant course matter was provided",
                                "Recommended useful textbooks, websites, periodicals","Syllabus was substantially covered in the class",
                                "Number of worked examples and tutorials were adequate","The lecturer promotes self studies by the student",
                                "Practical applications relevant to the module were discussed","Professional approach of the lecturer was high",
                                "The lecturer advised regarding evaluation","Continuous assessment helps the learning process",
                                "Feedback on continuous assessment was helpful to identify my weaknesses before he final examination"]
        for i in task_org_responses:
            sent.append(sent)
            print (i)
            sent = input()
            sent.appned(sent)
      

    # superstate - Ask about student interaction
    def student_interaction():
        # superstate - Student Interaction
        student_int_responses = ["To what extent did the lecturer facilitate and encourage active discussions among students?", "How effective was the lecturer in creating an environment conducive to student participation?",
                                 "How frequently did the lecturer recognize and appreciate student responses?", "How well did the lecturer manage the balance between encouraging participation and maintaining order?",
                                 "How actively did the lecturer encourage students to ask questions?", "How constructive and helpful was the feedback given to students during discussions?",
                                 "How well did the lecturer adapt their teaching based on student responses?", "What strategies did the lecturer employ to enhance student engagement during discussions?",
                                 "How effective were these strategies in promoting a dynamic and engaging learning environment?", "ow accessible and approachable did the lecturer appear during interactions with students?",
                                 "How did the lecturer balance individual and group interactions for good learning experience?"]
        for i in student_int_responses:
            sent.append(sent)
            print (i)
            sent = input()
            sent.appned(sent)
        
    
    # superstate - Ask about Clarity 
    def clarity():
        # superstate ask about Clarity of the lecture
        clarity_responses = ["Was the pace of the delivery consistent throughout the session?", "Did the lecturer maintain an appropriate speed, allowing for easy understanding?",
                             "How effective were the black/white board or PowerPoint presentations in conveying information?", "Could you easily hear and understand the lecturer throughout the session?",
                             "How clear were the verbal explanations provided by the lecturer?", "Were complex concepts articulated in a way that was easy to comprehend?",
                             "How effectively did the lecturer address questions to ensure understanding?", "How responsive was the lecturer to feedback provided by students?",
                             "How well did the lecturer accommodate diverse learning preferences?"]
        for i in clarity_responses:
            sent.append(sent)
            print (i)
            sent = input()
            sent.appned(sent)


    # listeing to user
    def listening_to_user(sent):
        listening_response =["I agree with your opinion", "Please share more about your opinion, that will be useful for future students",
                             "feel free to talk about any inconvenience you have"]
        for i in listening_response:
            print (i)

    # supporting to user
    def Supporting_user(sent):
        supporting_user =["Please don't miss out any lectures each and every lectures will be valuable", "At least you will learn something in the lecture rather than avoiding it",
                          "In lecutres try to concentrate on what is explained", "You can view the lecture materials before the lecture is conducted then you can understand more on lecutre itself"]
        for i in supporting_user:
            print (i)


    # Main state - Concluding
    def concluding():
        concluding_responses = ["Thank you for your insights and valuable input. Your perspective is greatly appreciated", "I appreciate your time and thoughtful responses. Thank you for sharing your thoughts with me",
                                "Thank you for taking the time to discuss these matters with me. Your input has been instrumental", "I'm grateful for the opportunity to hear your thoughts on this. Thank you for your time and thoughtful contributions",
                                "Your time and expertise are highly valued. Thank you for engaging in this discussion with me", "Thank you for spending your valuable time sharing your experiences and opinions. It has been a meaningful conversation",
                                "I want to express my gratitude for your time and the enriching conversation we've had. Thank you for your insights", "Thank you for your thoughtful responses. Your time and input have been invaluable to this discussion",
                                "I appreciate the depth of our conversation. Thank you for dedicating your time and sharing your perspective with me", "Thank you for your time and openness in discussing these matters. It has been a pleasure engaging in this conversation with you"]
        print (concluding_responses[rn.randint(0,len(concluding_responses)-1)])


    # calulating the fuzzy output value considering emotional cue and user engagement as input
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
    