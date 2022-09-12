# Resume-Screening-Application
## The Project folder contains the python files to be executed 
## 1 PreProcessing.py -  
pre processing the resumes and dataset splitting, 
the processed data is pickled:
Kaggle_labeled_dataframe.pickle - resumes from kaggle processed and stored as dataframe
labeled_dataframe.pickle - sample resumes are processed and pickled 
Final_labeled_dataframe.pickle - dataframe which all the available resumes as rows and features as columns
the data set is split and pickled into train , validation and test sets - 6 .pickle files
## 2 Training_voted.py:
uses an ensemble of classifiers and trains it against the training data
pickles the trained model - vote_clf.pickle
pickles the features to be used while testing ( for new unseen resumes) - Features.pickle

## 3 Scoring_Output.py :
takes a path to the resumes as an input and returns the output as a dataframe

## app.py :
renders the htmal templates *
stores the resumes in a temporary folder named "Files" **
gives a .csv file as output

## .env :
stores the path of the upload folder i.e. "Files"

*activate the virtual environment in the directory where Project folder is placed
** create an empty folder named "Files" in Project folder

## pickled folder:
stores all the pickle files mentioned above
