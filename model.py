
# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn



data1=pd.read_csv('./data_csv.csv')
data2=pd.read_csv('toddler.csv')
data3=pd.read_csv('autism_screening.csv')



df1=pd.concat([data1.iloc[:,1:11],data1.iloc[:,[12,22,23,24,25,26,27]]],axis=1)


df2=pd.concat([data2.iloc[:,1:12],data2.iloc[:,13:]],axis=1)
df2['Age_Mons']=(df2['Age_Mons']/12).astype(int)


df3=pd.concat([data3.iloc[:,0:15],data3.iloc[:,-2:]],axis=1)


order_test= pd.DataFrame({
    'df1': df1.columns,
    'df2': df2.columns ,
    'df3': df3.columns
})
order_test

df2.columns = df3.columns = df1.columns
data_fin = pd.concat([df3, df2, df1], axis=0)


object_cols = data_fin.select_dtypes('O').columns

object_df = pd.DataFrame({
    'Objects': object_cols,
    'Unique values': [data_fin[col].unique() for col in object_cols],
    'number of unique values':[data_fin[col].nunique()for col in object_cols]
})

object_df

replacements = {
    'f': 'F',
    'm': 'M',
}
data_fin['Sex'] = data_fin['Sex'].replace(replacements)

replacements = {
    'yes': 'Yes',
    'no': 'No',
}
data_fin['Jaundice'] = data_fin['Jaundice'].replace(replacements)

replacements = {
    'yes': 'Yes',
    'no': 'No',
}
data_fin['Family_mem_with_ASD'] = data_fin['Family_mem_with_ASD'].replace(replacements)

replacements = {
    'YES': 'Yes',
    'NO': 'No',
}
data_fin['ASD_traits'] = data_fin['ASD_traits'].replace(replacements)

replacements = {
    'middle eastern': 'Middle Eastern',
    'Middle Eastern ': 'Middle Eastern',
    'mixed': 'Mixed',
    'asian': 'Asian',
    'black': 'Black',
    'south asian': 'South Asian',
    'PaciFica':'Pacifica',
    'Pasifika':'Pacifica'

}
data_fin['Ethnicity'] = data_fin['Ethnicity'].replace(replacements)

replacements = {
    'Health care professional':'Health Care Professional',
    'family member':'Family Member',
    'Family member':'Family Member'
}
data_fin['Who_completed_the_test'] = data_fin['Who_completed_the_test'].replace(replacements)

object_cols = data_fin.select_dtypes('O').columns

object_df = pd.DataFrame({
    'Objects': object_cols,
    'Unique values': [data_fin[col].unique() for col in object_cols],
    'number of unique values':[data_fin[col].nunique()for col in object_cols]
})

object_df

data_fin['Ethnicity'].replace('?', np.nan, inplace=True)
data_fin['Who_completed_the_test'].replace('?', np.nan, inplace=True)
pd.DataFrame(data_fin.isnull().sum(),
             columns=["Missing Values"]).style.bar(color = "#84A9AC")

idf=data_fin.copy()
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='most_frequent')
imputed_data = pd.DataFrame(imp.fit_transform(idf))
imputed_data.columns = idf.columns
imputed_data.index = idf.index

pd.DataFrame(imputed_data.isnull().sum(),
             columns=["Missing Values"]).style.bar(color = "#84A9AC")

data = imputed_data.copy()



data_n= data.drop(columns = ['Ethnicity','Who_completed_the_test'])

data_n.head()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(max_depth = 10, random_state=0)

data_n.head()

data_n['Sex'].replace({"M":1, "F":0}, inplace = True)
data_n['Jaundice'].replace({"Yes":1, "No":0}, inplace = True)
data_n['Family_mem_with_ASD'].replace({"Yes":1, "No":0}, inplace = True)
data_n.head()

y = data_n['ASD_traits']
x = data_n.drop(columns = ['ASD_traits'])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=12, test_size=0.5)

X_train.shape, X_test.shape



# Random Forest
rfc.fit(X_train, Y_train)
Y_pred_rfc = rfc.predict(X_test)


#Make pickle file of our model
pickle.dump(rfc, open("model.pkl","wb"))



# features=np.array([[1,1,0,0,0,1,1,0,0,0,3,1,1,1]])
# p=rfc.predict(features)
# print("autisum detection :", p)