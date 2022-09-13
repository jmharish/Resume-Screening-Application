from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn import svm 
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter
from pyresparser import ResumeParser
import pickle
from decouple import config
from sklearn.ensemble import  VotingClassifier as vc
import PyPDF2 as pf
from fpdf import FPDF
import textract
from cmath import nan 


path = config("paths")

def test(p):
    pkl_file = open("C:\Harish\iGreenData_internship\pickled\Features.pickle","rb") ## change path accordingly
    lf = pickle.load(pkl_file) # list of features, check if they are there in resume or not
    pkl_file.close()

    os.chdir(p)#location of resumes
    l = os.listdir(p)
    L_ser =[]
    L_index = []
    L_proj_skills = []
    for i in l:
        di =  ResumeParser(i).get_extracted_data()
        d = dict({})
        s = set({})
        for ic in di['skills']:
            s.add(ic.lower())
        for j in lf:
            if j in s:
                d[j] = 1
            else:
                d[j] = 0
        s = pd.Series(d)
        L_ser.append(s)
        L_index.append(i)
        if i.endswith(".pdf"):
            L_proj_skills.append(pdf_ext(i))
        elif i.endswith(".docx"):
            L_proj_skills.append(DOC_ext(i))
        
        
    
    
    df2 = pd.DataFrame(L_ser,index = L_index) #the resume skills are extractedand a dataframe is created 
    df2.fillna(0,inplace = True)
    

    pkl_file = open("C:\Harish\iGreenData_internship\pickled\Voted_clf.pickle","rb") ## change path accordingly
    vote_clf = pickle.load(pkl_file)
    pkl_file.close()

    




    ##########################################################
    
    l_prob = []
    l_prob = vote_clf.predict_proba(df2)#
    df = pd.DataFrame()
    
    l = ['Automation Testing', 'Blockchain', 'Business Analyst', 'Civil Engineer', 'Data_Analyst', 'Database', 'DevOps', 'DotNet Developer', 'ETL Developer', 'Electrical Engineering', 'Hadoop', 'Java', 'Mechanical Engineer', 'Network Security Engineer', 'Operations Manager', 'Python Developer', 'SAP Developer', 'SRE', 'Testing', 'Web Designing']
    for j in range(len(l)) :
        df[l[j]] = [i[j]*100 for i in l_prob]
    l_lbl = list(vote_clf.predict(df2))
    ma_prob =0
    mi_prob=0
    avg_prob =0
    k =[]
    for i in l_prob:
        k.append(max(i*100))
    ma_prob = max(k)
    mi_prob = min(k)
    avg_prob = sum(k)/len(k)
    cols = df.columns.tolist()
    df["Predicted Label"] = l_lbl 
    df["Probablity of the Predicted Label"] = k
    df["Name"] = list(df2.index)
    #changing order of columns in csv 
    new_cols  = ["Name","Predicted Label","Probablity of the Predicted Label"]+cols
    df = df[new_cols]   #predicted label colummn is next to the name of the resume
    df["Skills used in Projects"] = L_proj_skills

    
    return df
    
    """fid = open("C:\Harish\iGreenData_internship\pickled\Output.csv","w")
    df.to_csv(fid)  #stored in the csv file
    fid.close()
    c = Counter(df['Predicted Label'])
    dic = dict(c)
    
    label_count =[dic[i] for i in list(set(df['Predicted Label']))]
    pie_label = [i+" "+str(dic[i]) for i in list(set(df['Predicted Label']))]
    plt.figure()
    plt.pie(label_count,labels=pie_label)
    fid = open("C:\Harish\iGreenData_internship\pickled\Output_chart.png","wb") ## change path accordingly
    plt.savefig(fid)  #stores chart as png file
    fid.close()"""
     
    
    #print("MAXIMUM PREDICTION PROBABLITY OF TRUE LABEL:",ma_prob)
    #print("MINIMUM PREDICTION PROBABLITY OF TRUE LABEL:",mi_prob)
    #print("AVERAGE PREDICTION PROBABLITY OF TRUE LABEL:",avg_prob)


def pdf_ext(filename):
    fid = open(filename,"rb")
    pdf = pf.PdfFileReader(fid)
    no_of_pages = pdf.getNumPages()
    
    txt = ""
    for i in range(no_of_pages):
        txt = txt+pdf.getPage(i).extractText()
        
    del pdf
    fid.close()
    raw_txt = txt

    txt = txt.splitlines()


    lp = []
    lt = []
    for i in txt:
        if i.istitle() :# finds if the line is a heading or title
            lt.append(i)
            if "Project"in i or "project" in i or "PROJECT" in i or "WORK" in i or "Work" in i or "work" in i or "Responsibilities" in i or "RESPONSIBILITIES" in i or "responsibilities" in i or "ENGINEER" in i or "Engineer" in i or "engineer" in i or "PRIVATE" in i or "Private" in i or "private" in i or "LIMITED" in i or "Limited" in i or "limited" in i or "PVT" in i or "Pvt" in i or "pvt" in i or "ROLES" in i or"Roles" in i or "roles" in i:
                lp.append(i)
                

    if(len(lp)==0):
        return 0
    next_title_ind  = lt.index(lp[len(lp)-1])+1 # to check if there are headings after project
    proj ="" # stores the text contained within the project section alone
    proj = raw_txt[raw_txt.index(lp[0]):len(raw_txt)]
    


    fid = open("Temp1.txt","w+")
    fid.truncate(0)
    fid.close()



    fid = open("Temp1.txt","w+")
    fid.write(proj.encode('utf8').decode('ascii', 'ignore')) #writing the project section of the resume 
    fid.close()

    pdf = FPDF()   #creating a pdf file adding page and setting font for writing the text into the pdf 
    pdf.add_page()
    pdf.set_font("Arial", size = 5)
    f = open("Temp1.txt","r")

    for x in f.readlines():
        pdf.cell(0,5, txt = x.encode('windows-1252').decode('ascii', 'ignore'), ln = 1, align = 'L')
    f.close()
    pdf.output("Temp.pdf")

    di =  ResumeParser("Temp.pdf").get_extracted_data()
    #the Temp pdf is used to extract the skills within the project section 


    

    
    os.remove("Temp1.txt")
    os.remove("Temp.pdf")
    os.remove(filename)
    return(di['skills'])


def DOC_ext(filename):
    t  = textract.process(filename)
    txt = str(t)
    raw_txt = txt
    txt = txt.split(r"\n")
    
    lp = []
    lt = []
    for i in txt:
        if i.istitle() :# finds if the line is a heading or title
            lt.append(i)
            if "Project"in i or "project" in i or "PROJECT" in i or "WORK" in i or "Work" in i or "work" in i or "Responsibilities" in i or "RESPONSIBILITIES" in i or "responsibilities" in i or "ENGINEER" in i or "Engineer" in i or "engineer" in i or "PRIVATE" in i or "Private" in i or "private" in i or "LIMITED" in i or "Limited" in i or "limited" in i or "PVT" in i or "Pvt" in i or "pvt" in i or "ROLES" in i or"Roles" in i or "roles" in i:
                lp.append(i)
                

    if(len(lp)==0):
        return 0
    next_title_ind  = lt.index(lp[len(lp)-1])+1 # to check if there are headings after project
    proj ="" # stores the text contained within the project section alone
    proj = raw_txt[raw_txt.index(lp[0]):len(raw_txt)]
    
    

    fid = open("Temp1.txt","w+")
    fid.truncate(0)
    fid.close()



    fid = open("Temp1.txt","w+")
    fid.write(proj.encode('utf8').decode('ascii', 'ignore')) #writing the project section of the resume 
    fid.close()

    pdf = FPDF()   #creating a pdf file adding page and setting font for writing the text into the pdf 
    pdf.add_page()
    pdf.set_font("Arial", size = 5)
    f = open("Temp1.txt","r")

    for x in f.readlines():
        pdf.cell(0,5, txt = x.encode('windows-1252').decode('ascii', 'ignore'), ln = 1, align = 'L')
    f.close()
    pdf.output("Temp.pdf")

    di =  ResumeParser("Temp.pdf").get_extracted_data()
    #the Temp pdf is used to extract the skills within the project section 


    
    os.remove("Temp1.txt")
    os.remove("Temp.pdf")
    os.remove(filename)
    return di['skills']






