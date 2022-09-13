import os 
import pandas as pd
from pyresparser import ResumeParser
import pickle
from fpdf import FPDF

####################### actual resumes are extrcated ###################
"""os.chdir("C:\Harish\iGreenData_internship\Dummy Resumes - AI_ML Internship\labeled_resumes")## change path accordingly to location of  input resumes
l = os.listdir("C:\Harish\iGreenData_internship\Dummy Resumes - AI_ML Internship\labeled_resumes") ## change path accordingly same as above path
L_ser =[]
L_labels = []
L_index = []
for i in l:
    di =  ResumeParser(i).get_extracted_data()
    d = dict({})
    for ic in di['skills']:
        d[ic.lower()] = True
    s = pd.Series(d)
    L_ser.append(s)
    L_index.append(i)

    if i.startswith("Java") or i.startswith("java") :
        L_labels.append("Java")
    elif i.startswith("Data")  :
        L_labels.append("Data_Analyst")
    elif i.startswith("DevOps")  :
        L_labels.append("DevOps")
    elif i.startswith("SRE")  :
        L_labels.append("SRE")
    elif i.startswith("Automation Testing")  :
        L_labels.append("Automation Testing")
    
    
    

df2 = pd.DataFrame(L_ser,index = L_index)

df2["Label"] = L_labels

pkl_file = open("C:\Harish\iGreenData_internship\pickled\labeled_dataframe.pickle","wb") ## change path accordingly to pickled folder location
pickle.dump(df2,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\labeled_dataframe.pickle","rb") ## change path accordingly to pickled folder location
df2 = pickle.load(pkl_file)
pkl_file.close()


print(len(df2))
print(set(df2["Label"]))
"""
"""######### Kaggle resumes are extracted #################################
os.chdir("C:\Harish\iGreenData_internship\Kaggle Dataset") ## change path according to  location of Kaggle dataset 
df1 = pd.read_csv("UpdatedResumeDataSet.csv") ## give the kaggle dataset folder name
L_index = []
L_ser =[]
L_labels = list(df1["Category"])


for i in range(len(df1["Resume"])):  
    fid = open("Temp1.txt","w+")
    fid.truncate(0)
    fid.close()



    fid = open("Temp1.txt","w+")
    fid.write(df1["Resume"][i].encode('utf8').decode('ascii', 'ignore')) #writing the text of the resume onto a text file
    fid.close()


    pdf = FPDF()   #creating a pdf file adding page and setting font for writing the text into the pdf 
    pdf.add_page()
    pdf.set_font("Arial", size = 5)
    f = open("Temp1.txt","r")
    
    for x in f.readlines():
        pdf.cell(0,5, txt = x.encode('windows-1252').decode('ascii', 'ignore'), ln = 1, align = 'L')
    f.close()
    pdf.output("Temp.pdf")

    di =  ResumeParser("C:\Harish\iGreenData_internship\Kaggle Dataset\Temp.pdf").get_extracted_data() ## change path accordingly to any location for a temporary file
    d = dict({})
    for ic in di['skills']:
        d[ic.lower()] = True
    s = pd.Series(d)
    L_ser.append(s)
    L_index.append(L_labels[i] + str(i))
    
    
df = pd.DataFrame(L_ser, index=L_index)
df["Label"] = L_labels



pkl_file = open("C:\Harish\iGreenData_internship\pickled\Kaggle_labeled_dataframe.pickle","wb") ## change path accordingly pickled folder location
pickle.dump(df,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Kaggle_labeled_dataframe.pickle","rb") ## change path accordingly pickled folder location
df = pickle.load(pkl_file)
pkl_file.close()
d = pd.DataFrame()

cf = df

cf.drop(cf[cf["Label"]=="HR"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Arts"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Advocate"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Sales"].index , inplace = True)
cf.drop(cf[cf["Label"]=="Health and fitness"].index , inplace = True)
cf.drop(cf[cf["Label"]=="PMO"].index , inplace = True)

# Java Developer to Java
c= cf[cf["Label"]=="Java Developer"]
c.drop(columns = "Label",inplace = True)
c["Label"] = "Java"
cf.drop(cf[cf["Label"]=="Java Developer"].index , inplace = True)
cf = cf.append(c,ignore_index = False )

#Data Science to Data_Analyst
c= cf[cf["Label"]=="Data Science"]
c.drop(columns = "Label",inplace = True)
c["Label"] = "Data_Analyst"
cf.drop(cf[cf["Label"]=="Data Science"].index , inplace = True)
cf = cf.append(c,ignore_index = False )

#Devops Engineer to DevOps
c= cf[cf["Label"]=="DevOps Engineer"]
c.drop(columns = "Label",inplace = True)
c["Label"] = "DevOps"
cf.drop(cf[cf["Label"]=="DevOps Engineer"].index , inplace = True)
cf = cf.append(c,ignore_index = False )




pkl_file = open("C:\Harish\iGreenData_internship\pickled\labeled_dataframe.pickle","rb") ## change path accordingly pickled folder location
dfb = pickle.load(pkl_file)
pkl_file.close()

print(set(dfb["Label"]))

x = len(dfb["Label"])

print(x)


 


dfb = dfb.append(cf,ignore_index = False)
dfb.fillna(False,inplace = True)
dfb.replace(True,1,inplace = True)
dfb.replace(False,0,inplace = True)
pkl_file = open("C:\Harish\iGreenData_internship\pickled\Final_labeled_dataframe.pickle","wb") ## change path accordingly to pickled folder location
pickle.dump(dfb,pkl_file)
pkl_file.close()
print(dfb.head())
unique_lbl = set([i for i in list(dfb["Label"])])
x = len(dfb["Label"])

print(unique_lbl)"""


####### Data set Splitting ######################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Final_labeled_dataframe.pickle","rb") ## change path accordingly to pickled folder location
df = pickle.load(pkl_file)
pkl_file.close()







print(set(df["Label"]))

Y = df["Label"]
X = df.drop(columns = ["Label"])
print( "number of labels:######",len(list(set(df["Label"]))))
skf5 = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf4 = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
for train_index, test_index in skf5.split(X, Y):  # test set gets 20% of dataset (1/5)
    X_train_val, x_test = X.iloc[train_index], X.iloc[test_index]
    Y_train_val, y_test = Y.iloc[train_index], Y.iloc[test_index]

for train_index, test_index in skf4.split(X_train_val , Y_train_val ): # val. set gets 25%(1/4) of 80%(->X_train_val) of dataset i.e 20% of dataset
    X_train, x_val = X.iloc[train_index], X.iloc[test_index]
    Y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]



print(" train lenghts are:",len(X_train)," " ,len(Y_train))
print(X_train[0:5])
print(Y_train[0:5])







print(" testing lenghts are:",len(x_test)," " ,len(y_test))
print(x_test[0:5])
print(y_test[0:5])

print(" validation lenghts are:",len(x_val)," " ,len(y_val))
print(x_val[0:5])
print(y_val[0:5])

plt.figure()
plt.subplot(3,1,1)
c = Counter(Y_train)
d = dict(c)
x_axis = list(set(df["Label"]))
y_axis = [d[i] for i in x_axis]

plt.scatter(x_axis,y_axis)
t = plt.xticks([])
plt.title("Train Set")

plt.subplot(3,1,2)
c = Counter(y_val)
d = dict(c)
x_axis = list(set(df["Label"]))
y_axis = [d[i] for i in x_axis]

plt.scatter(x_axis,y_axis)
t = plt.xticks([]) #removes labels in the x axis
plt.title("Validation Set")

plt.subplot(3,1,3)
c = Counter(y_val)
d = dict(c)
x_axis = list(set(df["Label"]))
y_axis = [d[i] for i in x_axis]

plt.scatter(x_axis,y_axis)
t = plt.xticks(rotation= 90) #rotates the labels in x axis by 90 degrees
plt.title("Test Set")

plt.show()

print(df.head())


#pickling the train sets 
pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Train.pickle","wb") ## change path accordingly to pickled folder location
pickle.dump(X_train,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Train.pickle","wb")  ## change path accordingly to pickled folder location
pickle.dump(Y_train,pkl_file)
pkl_file.close()

# pickling the validation sets 
pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Val.pickle","wb")  ## change path accordingly to pickled folder location
pickle.dump(x_val,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Val.pickle","wb")  ## change path accordingly to pickled folder location
pickle.dump(y_val,pkl_file)
pkl_file.close()

#pickling test sets

pkl_file = open("C:\Harish\iGreenData_internship\pickled\X_Test.pickle","wb")  ## change path accordingly to pickled folder location
pickle.dump(x_test,pkl_file)
pkl_file.close()

pkl_file = open("C:\Harish\iGreenData_internship\pickled\Y_Test.pickle","wb")  ## change path accordingly to pickled folder location
pickle.dump(y_test,pkl_file)
pkl_file.close()

print("pickling done")


