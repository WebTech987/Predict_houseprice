# Predict_houseprice 
# house price prediction based on the boston data available on sklearn

## Softwares And Tools Required 
1. [Github Account](https://github.com)
2. [Heroku Account](https://signup.heroku.com/)
3. [VS code IDE](https://code.visualstudio.com/)
4. [Git CLI](https://git-scm.com/download/win)
5. [Anaconda](https://www.anaconda.com/products/distribution)
6. [Python Programming](https://www.tutorialgateway.org/python-programming-examples/)

create a new environment
'''
conda create -p venv python==3.9 -y
'''

setting up the user criterias to dumpt the files into github
use the command to set up user and gmail to commit the repository directly into our git-hub account
Note: remember to use the same email that has been used in github account
'''
git config --global user.name 
git config --global user.email
'''
## commands to dump the files 
To dump/add a single file
'''
git add file_name  EX: git add requirements.txt
'''
To dump/add all the files together
'''
git add .
'''

sometime when we install a new vscode and tries to run "git add ." command we might face an fatal error : "fatal : detected dubious ownership in repository at <file_name>.

if you face above mentioned fatal error : please run the below command else skip it if don't get fatal error :
'''
git config --global --add safe.drectory "*"
'''
After running the previous command run the elements add command to upload the files in the git
'''
git add .
'''

Then run the command to update the git 
'''
git update
'''

once we added the files at the git, run the below command to commit the files 
'''
git commit -m "initial commit"
'''

after all now we are ready to upload the files from local folder - git onto github repository
run the below command 
'''
git push
'''
Apop-up window will get open - Please enter password to login to into your github account 
***congratulations we successfully uploaded the all the files onto github repository***
