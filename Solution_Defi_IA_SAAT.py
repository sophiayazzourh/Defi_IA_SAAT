#!/usr/bin/env python
# coding: utf-8

# # Défi IA, solution proposée par la team SAAT

# # 1. Importation des librairies, données & scripts 

# ## 1.1 Librairies

import pandas as pd
import numpy as np
import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import sklearn.metrics as smet
import sklearn.model_selection as sms


# ## 1.2 Scripts 

import sys
sys.path.append('./scripts')
import Cleaning as ct
import Vectorization as Vecto 
import Learning as RL


# ## 1.3 Données

DATA_PATH = "./data"
train_df = pd.read_json(DATA_PATH+"/train.json") # Training data 
test_df = pd.read_json(DATA_PATH+"/test.json") # Testing data 
names = pd.read_csv(DATA_PATH+ '/categories_string.csv')['0'].to_dict()
jobs = pd.read_csv(DATA_PATH+'/train_label.csv', index_col='Id')['Category']
jobs = jobs.map(names)
jobs = jobs.rename('job') # The jobs of trainging data
genders = pd.read_json(DATA_PATH+'/train.json').set_index('Id')['gender']
# genders of the people in training data
train_label = pd.read_csv(DATA_PATH+"/train_label.csv") # the jobs numbered from 1 to 28

print("Libraries, Scripts & Data loaded")

# # 2. Pre-processing 

print("Pre-processing start at " + time.ctime()[11:19])
# ## 2.1 Minuscules

train_df["description_lower"] = [x.lower() for x in train_df.description]
test_df["description_lower"] = [x.lower() for x in test_df.description]


# ## 2.2 Cleaning 


ct.clean_df_column(train_df, "description_lower", "description_cleaned")
train_df[["description_lower", "description_cleaned"]] # Cleaning lower description in the train data

ct.clean_df_column(test_df, "description", "description_cleaned")
test_df[["description_lower", "description_cleaned"]] # Cleaning lower description in the test data


# ## 2.3 Vectorization by TFidf


X_test=test_df # Test data
X=train_df # Data to train and create the best model
y=train_label.Category.values # The reponse of the train data


X_train, X_valid, y_train, y_valid = sms.train_test_split(X, y, test_size=0.2, random_state=1)
# Devide the train data and reponse to the train and valid (X,y) (data,reponse) which helps us
# find out the best model

features_parameters = [[None, "count"],
                      [10000, "count"],
                      [None, "tfidf"],
                      [10000, "tfidf"],]


metadata = {}
for nb_hash, vectorizer_type in features_parameters:
    vect_method = Vecto.Vectorizer(vectorizer_type = vectorizer_type, nb_hash = nb_hash )
    ts = time.time()
    vec, feathash, X_train_vec = vect_method.vectorizer_train(X_train, columns = "description_cleaned")
    X_valid_vec = vect_method.apply_vectorizer(X_valid, columns = "description_cleaned", vec = vec, feathash = feathash)
    X_test_vec = vect_method.apply_vectorizer(X_test, columns = "description_cleaned", vec = vec, feathash = feathash)
    
    te = time.time()
    
    metadata.update({(nb_hash, vectorizer_type):te-ts})
    
    print("nb_hash : " + str(nb_hash) + ", vectorizer_type : " + str(vectorizer_type))
    print("Runing time for vectorization : %.1f seconds" %( metadata[(nb_hash, vectorizer_type)]))
    print("Test shape : " + str(X_test_vec.shape))
    print("Train shape : " + str(X_train_vec.shape))
    print("Valid shape : " + str(X_valid_vec.shape))

    vect_method.save_dataframe(X_test_vec, "test") # Vectorized X_test
    vect_method.save_dataframe(X_train_vec, "train") # Vectorized X_train
    vect_method.save_dataframe(X_valid_vec, "valid") # Vectorized X_valid
    
print("Pre-processing end at " + time.ctime()[11:19])

# # 3. Régression Logistique

print("Logistic Regression training start at " + time.ctime()[11:19])
# ## 3.1 Training 

FORCE_TO_RUN = True
features_parameters = [[None, "tfidf"]] # Using the TF-IDF (the more complicated method)

model_parameters = [["lr", {"C":[0.1, 1, 10]}]]

if FORCE_TO_RUN:
    metadata = {}
    for nb_hash, vectorizer_type in features_parameters:
        print(nb_hash, vectorizer_type)
        vect_method = Vecto.Vectorizer(vectorizer_type = vectorizer_type, nb_hash = nb_hash )
        X_train = vect_method.load_dataframe("train")
        Y_train = y_train
        X_valid = vect_method.load_dataframe("valid")
        Y_valid = y_valid

        for ml_model_name, param_grid in model_parameters:
            ml_class = RL.MlModel(ml_model_name=ml_model_name, param_grid=param_grid)
            best_model, best_metadata = ml_class.train_all_parameters(X_train, Y_train, X_valid, Y_valid
                                                                      , save_metadata=False)
            accuracy_test = best_model.score(X_valid, Y_valid)
            f1_macro_score_test = smet.f1_score(best_model.predict(X_valid),Y_valid, average='macro')
            balanced_accuracy_test = smet.balanced_accuracy_score(best_model.predict(X_valid),Y_valid)
            best_metadata.update({"balanced_accuracy_test":balanced_accuracy_test,"accuracy_test": accuracy_test, "f1_macro_score_test":f1_macro_score_test})
            metadata.update({(vectorizer_type, str(nb_hash), ml_model_name): best_metadata})

print("Logistic Regression training end at " + time.ctime()[11:19])

# ## 3.2 Prediction

print("Logistic Regression prediction start at " + time.ctime()[11:19])

X_test = vect_method.load_dataframe("test") # Loading the test data
y_pred = best_model.predict(X_test) # Predicting the response

print("Logistic Regression prediction end at " + time.ctime()[11:19])


# # 4. Génération des résultats pour Kaggle

test_df["Category"] = y_pred
baseline_file = test_df[["Id","Category"]]
if os.path.isdir('./results') == False:
    os.mkdir("./results")

baseline_file.to_csv("./results/baseline.csv", index=False)

print("Kaggle format results save in Results dir")

