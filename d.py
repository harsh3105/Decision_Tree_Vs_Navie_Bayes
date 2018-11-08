import numpy as np 
import pandas as pd
import string
import codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

comp={"Accuracy":0.0,"Precision":0.0,"Recall":0.0,"F-score":0.0}
comp1={"Accuracy":"nill","Precision":"nill","Recall":"nill","F-score":"nill"}
# Function importing Dataset 
def importdata():
    data = codecs.open("d.txt",encoding="utf-8").read()
    Y,X = [],[]
    for i,line in enumerate(data.split("\n")):
        content = line.split()
        #clear
        # print(content)
        Y.append(content[0])
        X.append(content[1])


    balance_data = pd.DataFrame()
    balance_data['text'] = X
    balance_data['label'] = Y


    #balance_data = pd.read_csv('balance-scale.csv', sep= ',', header = None)
    #print(balance_data)
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
  
# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    """X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 0]
    print(X)"""
    
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    balance_data['text'], balance_data['label'], test_size = 0.3, random_state = 100)


    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    print("\n\nencoder\n",y_train)


    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(balance_data['text'])

    xtrain_count =  count_vect.transform(X_train)
    xvalid_count =  count_vect.transform(X_test)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(balance_data['text'])
    X_train_tfidf =  tfidf_vect.transform(X_train)
    X_test_tfidf =  tfidf_vect.transform(X_test)




      
    return  X_train_tfidf, X_test_tfidf, y_train, y_test 
      
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train_tfidf, X_test_tfidf, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train_tfidf, y_train) 
    return clf_entropy 

# Function to perform training with multinomialNB
def tarin_using_multinomialNB(X_train_tfidf, X_test_tfidf, y_train):
    mnb = MultinomialNB(alpha=0.5,fit_prior=True)
    mnb.fit(X_train_tfidf, y_train)
    return mnb
  
# Function to make predictions 
def prediction(X_test_tfidf, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test_tfidf) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred

      
# Function to calculate accuracy 
def cal_accuracy(a,y_test, y_pred): 
    
    print("Confusion Matrix") 
    print(confusion_matrix(y_test, y_pred))
     
    print ("Accuracy") 
    asc=accuracy_score(y_test,y_pred)*100
    print(asc)
    if comp["Accuracy"]<=asc:
        comp["Accuracy"]=asc
        comp1["Accuracy"]=a
      
    print("Report") 
    print(classification_report(y_test, y_pred))

    print("Report 1")
    precision,recall,fscore,support=score(y_test,y_pred,average='micro')
    if comp["Precision"]<=precision:
        comp["Precision"]=precision
        comp1["Precision"]=a
    if comp["Recall"]<=recall:
        comp["Recall"]=recall
        comp1["Recall"]=a
    if comp["F-score"]<=fscore:
        comp["F-score"]=fscore
        comp1["F-score"]=a
  
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X_train_tfidf, X_test_tfidf, y_train, y_test = splitdataset(data)  
    clf_entropy = tarin_using_entropy(X_train_tfidf, X_test_tfidf, y_train)
    mnb=tarin_using_multinomialNB(X_train_tfidf, X_test_tfidf, y_train)
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test_tfidf, clf_entropy) 
    cal_accuracy("decision tree",y_test, y_pred_entropy)

    print("Results using multinomial NB:")
    #prediction using multinomial NB
    y_pred_NB=prediction(X_test_tfidf, mnb)
    cal_accuracy("Multninomial naive Bayes",y_test,y_pred_NB)
    print("results based on micro weights")
    for key,val in comp1.items():
        print (key,"=>",val)
      
# Calling main function 
if __name__=="__main__": 
    main() 
