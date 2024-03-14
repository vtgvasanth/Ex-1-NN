<H3>NAME : vasanth S</H3>
<H3>REGISTER NUMBER : 212222040175</H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py

#import libraries
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

```
```py

#Read the dataset from drive
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/NEURAL NETWORKS/Churn_Modelling.csv',index_col="RowNumber")
dataset.head()

```
```py

# Finding Missing Values
print(dataset.isnull().sum())

```
```py

#Handling Missing Values
dataset=dataset.drop(['Surname', 'Geography','Gender'], axis=1)

```
```py

#Check for Duplicates
dataset.duplicated().sum()

```
```py

#Normalize the dataset
scaler=StandardScaler()
dataset=pd.DataFrame(scaler.fit_transform(dataset))
dataset.head()

```
```py

#split the dataset into input and output
X,Y=dataset.iloc[:,:-1].values ,dataset.iloc[:,-1].values
print('Input:\n',X,'\nOutput:\n',Y)

```
```py

#splitting the data for training & Testing
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)
print("Xtrain:" ,len(Xtrain), "\nXtest:", len(Xtest))
print("\nYtrain:" ,len(Ytrain), "\nYtest:", len(Ytest))

```
## OUTPUT:

### READ THE DATASET : 
![Screenshot 2024-03-03 155233](https://github.com/vtgvasanth/Ex-1-NN/assets/128463280/01573c09-18de-40d9-b97a-c021c2d42a77)


### FINDING MISSING VALUES : 
![Screenshot 2024-03-03 155258](https://github.com/vtgvasanth/Ex-1-NN/assets/128463280/0c4dc9bc-c561-43a6-8977-c7ad0c6ea9b6)


### CHECK FOR DUPLICATES : 
![Screenshot 2024-03-03 155320](https://github.com/vtgvasanth/Ex-1-NN/assets/128463280/21392ff9-a089-4134-9335-7918ac8fce2c)


### NORMALIZE THE DATASET : 
![Screenshot 2024-03-03 155343](https://github.com/vtgvasanth/Ex-1-NN/assets/128463280/73bcc62c-b94c-4b68-ba29-ab64daf19a86)


### SPLITTING THE DATA FOR TRAINING AND TESTING : 
![Screenshot 2024-03-03 155413](https://github.com/vtgvasanth/Ex-1-NN/assets/128463280/9bc070bd-f3ac-4158-9a48-54264fa300f4)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
