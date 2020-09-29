from nltk.stem import PorterStemmer
from keras.models import Sequential,load_model
from keras.layers import Dense
import nltk
import numpy
import random
import json
import pickle


stemmer=PorterStemmer()
with open("intents1.json") as file:
    data=json.load(file)
try:
    with open("data.pickle","rb") as f:
        word,labels,training,output=pickle.load(f)
except:
    word=[]
    labels=[]
    docs_x=[]
    docs_y=[]
    for intent in data["intents"]:
        for patterns in intent["patterns"]:
            w=nltk.word_tokenize(patterns)
            word.extend(w)
            docs_x.append(w)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    word=[stemmer.stem(w.lower()) for w in word if w != "?"]
    word=sorted(list(set(word)))
    labels=sorted(labels)
    training=[]
    output=[]
    out_empty=[0 for _ in range(len(labels))]
    for x,doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(p) for p in doc]
        for w in word:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        out_row=out_empty[:]
        out_row[labels.index(docs_y[x])]=1

        training.append(bag)
        output.append(out_row)

    training=numpy.array(training)
    output=numpy.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((word,labels,training,output),f)

model=Sequential()
model.add(Dense(8,input_dim=len(training[0]),activation="softmax"))
model.add(Dense(8,activation="softmax"))
model.add(Dense(len(output[0]),activation="softmax"))
try:
    model=load_model("C:/Users/peedr/OneDrive/Desktop/machine learning/model1.h5")
except:
    model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])
    model.fit(training,output,epochs=10000,batch_size=8)
    model.save("model1.h5")
def bag_of_words(inp,word):
    bag=[[0 for _ in range(len(word))]]
    tok_inp=nltk.word_tokenize(inp)
    stem_inp=[stemmer.stem(z.lower())for z in tok_inp]
    for p in stem_inp:
        for i,x in enumerate(word):
            if x==p:
                bag[0][i]=1
    return numpy.array(bag)
def chat():
    print("To Quit enter bye")
    while True:
        inp=input("you: ")
        if inp.lower()=="bye":
            print("Robo:It was fun talking to you.See you later.")
            break
        result=model.predict(bag_of_words(inp,word))
        #print(result)
        tag=labels[numpy.argmax(result)]
        if max(max(result))>0.7:
            for tg in data["intents"]:
                if tg["tag"]==tag:
                    responses=tg["responses"]
            print("Robo:",random.choice(responses))
        else:
            print("Robo: Sorry, I don\'t understand:(")
chat()



        
