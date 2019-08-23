import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
# %% markdown
# My first go at some kind of competition using a RandomForestClassifier
# %%
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

X = pd.read_csv('train.csv',index_col='PassengerId')
X_test = pd.read_csv('test.csv', index_col='PassengerId')

X.head()

# now we should probably remove some columns like the name and ticket to make things easier for now
# like cabin should be dropped and ticket number doesn't really matter
# we will deal with the rest of the nulls later
drop_col = ['Ticket', 'Cabin', 'Name']
X = X.drop(drop_col, axis=1)
X_test = X_test.drop(drop_col, axis=1)

# print(X.columns)
# print(X.isnull().sum())
# embarked_null = X[X['Embarked'].isnull()].index.tolist()
# X = X.drop(embarked_null, axis=0)
# print(X.isnull().sum())

# now we should do some visualizations of the dataset
# we would like to know the distributions in some of these categories
Pclass = X.groupby('Pclass')
# print(Pclass['Survived'].sum())

Age = X.groupby('Age')
# print(Age['Survived'].sum())

plt.figure()
sns.catplot(x='Pclass', y='Survived', data=X, kind='bar')

sns.catplot(x='Pclass', y='Survived', data=X, kind='bar', hue='Sex')

sns.catplot(x='Embarked', y='Survived', data=X, kind='bar')

plt.figure()
sns.distplot(a=Age['Survived'].sum())

# %% markdown
# **TRENDS**
# * Pclass affects surivability 1 > 2 > 3
# * Females are much more likely to survive than males
# * People embarking from different places has an effect survivability C > Q > S
# %%
# we should also split X data into training and validation datasets
y = X.Survived
X = X.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
# %%
# know lets clean up the code accordingly so that we can input it into some model
# we want to make a pipeline
# First lets identify the categorical and numerical columns

cat_cols = []
num_cols = []

for col in X.columns:
    if X[col].dtype == 'object':
        cat_cols.append(col)
    else:
        num_cols.append(col)

print(cat_cols)
print(num_cols)

# should also deal with the null values in the Age column
from sklearn.impute import SimpleImputer
# my_imputer = SimpleImputer()

# imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X[['Age']]))
# imputed_X_valid = pd.DataFrame(my_imputer.transform(valid_X[['Age']]))

# X_train.loc[:, 'Age'] = imputed_X_train.loc[:, 0]
# X_valid.loc[:, 'Age'] = imputed_X_valid.loc[:, 0]

# print(X_valid.columns)
# %% markdown
# We could do all the preprocessing separately but it would be best to
# just write a pipeline so we don't have to keep typing it over and over again
# for each datset
# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# first we need to make the transformers for each kind of column
# in this case for the numerical and categorical columns

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder())
])

# we have our transformers so we need to make a ColumnTransformer
# object that uses these transformers

preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, num_cols),
    ('categorical', categorical_transformer, cat_cols)
])

# specify a model to use after applying these transformations
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Time to bundle up this preprcessor and our model together
my_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', model)
])
# %% markdown
# Time to test out the model with a RandomForestRegressor to see if it works.
# To use a different model, just update the model variable
# %%
from sklearn.metrics import accuracy_score
my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

score = accuracy_score(y_valid, preds)
print(score)

# %% markdown
# We need to predict our Test dataset to see what happens
# %%
preds2 = my_pipeline.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.index,
                       'Survived': preds2
})

output.to_csv('submission.csv', index=False)
# %%
