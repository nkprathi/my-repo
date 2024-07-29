import nltk 
from nltk.chat.util import Chat, reflections

# Define the pairs of patterns and responses
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how are you today?",]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there",]
    ],
    [
        r"what is your name?",
        ["I am a chatbot created by you.",]
    ],
    [
        r"how are you?",
        ["I'm doing good, how about you?",]
    ],
    [
        r"sorry (.*)",
        ["It's alright", "No problem",]
    ],
    [
        r"i'm (.*) doing good",
        ["Nice to hear that", "Alright, great!",]
    ],
    [
        r"(.*) age?",
        ["I'm a computer program, I don't have an age.",]
    ],
    [
        r"what (.*) want?",
        ["I just want to help you.",]
    ],
    [
        r"(.*) created you?",
        ["I was created by a brilliant developer.",]
    ],
    [
        r"(.*) (location|city)?",
        ["I'm in the cloud.",]
    ],
    [
        r"how (.*) weather (.*)?",
        ["I'm not sure, but I hope it's nice where you are!",]
    ],
    [
        r"quit",
        ["Bye, take care. See you soon!", "It was nice talking to you. Goodbye!"]
    ],
]

# Create the chatbot
def chatbot():
    print("Hi, I'm your chatbot. Type something to start a conversation (type 'quit' to stop)!")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    chatbot()