import flask
from flask import Flask , render_template , url_for , request ,Response
from werkzeug.wrappers import Response
from Scoring_Output import test , path
import pandas as pd
import os 
import io
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:\Harish\iGreenData_internship\Project\Files' ## change path accordingly

#the destination to store the uploaded files

    





@app.route("/upload")
def upload_file():
    return render_template("upload.html")# automaticaly looks in templates folder

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():

    basedir = os.path.abspath(os.path.dirname(__file__))

    #delete any files in upload folder before adding new files
    del_folder = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    os.chdir(del_folder)
    for i in os.listdir():
        os.remove(i)
    
    if request.method == 'POST':
        fs = request.files.getlist('files[]') # gets multiple files from requests
        # the parameter in getlist() or subscript of request.files[] must name of file type input in html
        """fs.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], fs.filename))"""
        for f in fs:
            f.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], f.filename))
            f.close()
            del f 
    
    
    

    
      # add the basedirectory and the path to upload folder and stores each file in upload folder
    
    return render_template("uploader.html")

      
@app.route('/download', methods = ['GET'])
def download():
    df = test(path)   # uses test function in Scoring_Output for generating output csv file 
    buffer = io.BytesIO()
    df.to_excel(buffer)# store the dataframe as excel using BytesIO object
    headers = {
    'Content-Disposition': 'attachment; filename=output.xlsx',
    'Content-type': 'application/vnd.ms-excel'
    }
    basedir = os.path.abspath(os.path.dirname(__file__))

    #delete any files in upload folder after the resumes are used
    del_folder = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    os.chdir(del_folder)
    for i in os.listdir():
        os.unlink(i)
    return Response(buffer.getvalue(), mimetype='application/vnd.ms-excel', headers=headers)
    # use the BytesIO object for creating a response (download file csv)  
if __name__ == "__main__":
    app.run(debug=True)
    
    
