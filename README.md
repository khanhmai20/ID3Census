# ID3 on Census Data Ananlysis


```python
import math
import pandas as pd
```

### Visualizing Data


```python
emails_df = pd.read_csv("assets/emails.csv")
playtennis_df = pd.read_csv("assets/playtennis.csv")
print(playtennis_df,"\n\n",emails_df)
```

        Day   Outlook Temperature Humidity    Wind PlayTennis
    0    D1     Sunny         Hot     High    Weak         No
    1    D2     Sunny         Hot     High  Strong         No
    2    D3  Overcast         Hot     High    Weak        Yes
    3    D4      Rain        Mild     High    Weak        Yes
    4    D5      Rain        Cool   Normal    Weak        Yes
    5    D6      Rain        Cool   Normal  Strong         No
    6    D7  Overcast        Cool   Normal  Strong        Yes
    7    D8     Sunny        Mild     High    Weak         No
    8    D9     Sunny        Cool   Normal    Weak        Yes
    9   D10      Rain        Mild   Normal    Weak        Yes
    10  D11     Sunny        Mild   Normal  Strong        Yes
    11  D12  Overcast        Mild     High  Strong        Yes
    12  D13  Overcast         Hot   Normal    Weak        Yes
    13  D14      Rain        Mild     High  Strong         No 
    
         ID  SUSPICIOUS WORDS  UNKNOWN SENDER  CONTAINS IMAGES CLASS
    0  376              True           False             True  spam
    1  489              True            True            False  spam
    2  541              True            True            False  spam
    3  693             False            True             True   ham
    4  782             False           False            False   ham
    5  976             False           False            False   ham


### Calculating entropy


```python
#pre: dataset as dataFrame object
#post: return the entropy from any dataFrame
def entropy(dataset):
    #Get the classification attribute of this dataset
    classifier = dataset.columns[-1]
    
    #Construct the dictionary to count the occurences of each unique outcome
    occurence = {}
    for outcome in list(dataset[classifier]):
        if outcome not in occurence:
            occurence[outcome] = 1
        else:
            occurence[outcome] += 1
    
    #Calculate entropy
    entropy = 0
    for value in occurence.values():
        total_outcome = len(dataset[classifier]) #Get the possible outcome of this classification
        probability = value/total_outcome
        entropy -= probability*math.log2(probability)
    
    return entropy
```

### Calculating Info Gain 


```python
#pre: dataset as dataFrame object
#post: return the info gain of all attributes
def infoGain(dataset,attribute):
    #Construct the dictionary to count the occurences of each unique value attribute
    occurence = {}
    for value in list(dataset[attribute]):
        if value not in occurence:
            occurence[value] = 1
        else: 
            occurence[value] += 1
    
    entropy_Attribute = 0
    
    #For each unique value, calculate the entropy of the classifier 
    for value in occurence: 
        sub_df = dataset[dataset[attribute] == value]
        sub_entropy = entropy(sub_df)
        probability = occurence[value]/len(dataset[attribute])
        entropy_Attribute += probability*sub_entropy
        
    #We have the following formular
    infoGain = entropy(dataset) - entropy_Attribute
    
    return infoGain   
```

### Testing infoGain on Playtennis Data


```python
#pre: take dataset as parameter
#post: return the maximun infoGain attribute

def gainfulAttr(dataset):
    attributes = list(dataset.columns)[0:-1] #Exclue the classifier attribute
    gain_attributes = {} #Key: Attribute, Value: InfoGain
    
    for attribute in attributes:
        if attribute not in gain_attributes:
            gain_attributes[attribute] = infoGain(dataset,attribute)
    
    decision = max(gain_attributes,key=gain_attributes.get)
    return decision
```


```python
gainfulAttr(playtennis_df.drop(columns=['Day']))
```




    'Outlook'



### Building ID3 Model


```python
#Building the ID3 model 
def id3Model(dataset,pruning,dictionary=None):
    
    #Threshold of penalize dataset when it reach this
    #return the most common classes.
    if len(dataset) < pruning:
        classifiers = list(dataset[dataset.columns[-1]].values)
        most_common = max(set(classifiers),key = classifiers.count)
        return most_common
    
    if dictionary is None: 
        dictionary ={}
        
    #If this is a pure set, all classes are the same, return the class value
    if (len(dataset[dataset.columns[-1]].unique())==1): #pure set
        return dataset[dataset.columns[-1]].unique()[0]
    
    #if no more attribute left
    #return the most common classes
    if (len(dataset.columns)==1): 
        classifiers = list(dataset[dataset.columns[-1]].values)
        most_common = max(set(classifiers),key = classifiers.count)
        return most_common
    
    #Recurse to build the tree
    else:
        #start with most gainful attribute attach it to the root node
        decision_Attr = gainfulAttr(dataset)
        dictionary[decision_Attr] = {} 
        unique_value = dataset[decision_Attr].unique()
        
        #iterate values in that attribute
        for value in list(unique_value):
            
            #partition the dataset
            sub_dataset = dataset[dataset[decision_Attr]==value]
            sub_dataset = sub_dataset.drop(columns=[decision_Attr])
            
            dictionary[decision_Attr][value] = id3Model(sub_dataset,pruning)
    
    return dictionary
```

### Visualize ID3 on PlayTennis and Email


```python
import pprint
emailModel = id3Model(emails_df.drop(columns=['ID']),0)
playtennisModel = id3Model(playtennis_df.drop(columns=['Day']),0)

census_df = pd.read_csv("assets/census_training.csv")
censusModel = id3Model(census_df,0)
pprint.pprint(emailModel)
print('\n')
pprint.pprint(playtennisModel)
```

    {'SUSPICIOUS WORDS': {False: 'ham', True: 'spam'}}
    
    
    {'Outlook': {'Overcast': 'Yes',
                 'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
                 'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}



```python
census_test_df = pd.read_csv("assets/census_training_test.csv")
```

### Build this Prediction on testing set


```python
#pre: id3 model, and the query of testing set. query = {attribute1: value1, attribute2: value2,...,}
#post: return the prediction of id3 output base on the query
def prediction(id3,query,default=1): 
    
    for key in list(query.keys()): 
        if key in list(id3.keys()):
            
            try:
                result = id3[key][query[key]] 
            except:
                return default 
            
            result = id3[key][query[key]]
            if isinstance(result,dict):
                return prediction(result,query)
            else:
                return result
```


```python
#pre : input dataset and the example ith
#post: building queries 

def queries(dataset,example): 
    query = {}
    for key in list(dataset.iloc[example][:].keys()):
        if key not in query: 
            query[key] = dataset.iloc[example].get(key)
    return query
```


```python
#pre : input id3 model and testing data
#post: testing the predictions on the dataset, output the statistics
def accuracy(id3,testing_data):
    correct = 0; 
    incorrect = 0; 
    classifier = testing_data.columns[-1]
    
    for example in range(0,len(testing_data)):
        query = queries(testing_data,example)
        
        predict = prediction(id3,query)
        if(query[classifier]==predict):
            correct += 1
        else:
            incorrect += 1
    
    print("Number of testing examples =\t",len(testing_data))
    print("Correct classifications =\t",correct)
    print("Incorrect classifications =\t",incorrect)
    print("Accuracy =\t\t ",correct/len(testing_data)*100,"%")
```

### Testing with no tree pruning


```python
accuracy(censusModel,census_test_df)
```

    Number of testing examples =	 15028
    Correct classifications =	 11289
    Incorrect classifications =	 3739
    Accuracy =		  75.11977641735427 %


### Testing with tree pruning


```python
censusModel_pruning = id3Model(census_df,30)
accuracy(censusModel_pruning,census_test_df)
```

    Number of testing examples =	 15028
    Correct classifications =	 11976
    Incorrect classifications =	 3052
    Accuracy =		  79.69124301304232 %

