from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
import xgboost as xgb
from sklearn.ensemble import  VotingClassifier
from pandas import DataFrame as df
from pandas import Series as s
import pickle

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","rb")
X_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","rb")
Y_Train = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_val.pickle","rb")
X_val = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_val.pickle","rb")
Y_val = pickle.load(pkl_file)
pkl_file.close()

X_Train = X_Train.append(X_val) #the validation set is also used to train the selected models
Y_Train = Y_Train.append(Y_val)
print(list(X_Train.columns))
 # the columns give the dimension or attributes using which a resume should be represented 

l = list(X_Train.columns)
pkl_file = open("C:\Harish\iGreenData_internship\pickled\Features.pickle","wb") 
pickle.dump(l,pkl_file)
pkl_file.close()

"""mNB_clf = MultinomialNB(alpha = 100)
bNB_clf = BernoulliNB(alpha = 100)
cNB_clf = ComplementNB(alpha = 100)
svm_clf = svm.SVC(C= 0.01,kernel ='linear',probability=True)
xg_clf = xgb.XGBClassifier( max_depth=4,reg_lambda = 3,eta = 0.3,gamma = 10 ) 
lr_clf = LogisticRegression(C = 0.05)"""
mNB_clf = MultinomialNB()
bNB_clf = BernoulliNB()
cNB_clf = ComplementNB()
svm_clf = svm.SVC(kernel ='linear',probability=True)
xg_clf = xgb.XGBClassifier( ) 
lr_clf = LogisticRegression()
kn =knn(weights="uniform")

models =[("mnb",mNB_clf), ("bnb",bNB_clf),('cnb',cNB_clf),('svm',svm_clf),('XG',xg_clf),("lr",lr_clf),('KNN',kn)]
vote_clf = VotingClassifier(estimators=models,voting='soft') #soft voting means using the prediction probailities of eac classifier
vote_clf = vote_clf.fit(X_Train,Y_Train)





pkl_file = open("C:\Harish\iGreenData_internship\pickled\Voted_clf.pickle","wb") 
pickle.dump(vote_clf,pkl_file)
pkl_file.close()

print("pickling done")
print(vote_clf.classes_)
