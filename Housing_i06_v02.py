import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

import time

# get the start time
st = time.time()

path = '..\Data\\'

housing = pd.read_csv(path+"housing-classification-iter6.csv")

X_test_compet = pd.read_csv(path+"test.csv")

X_test_compet.drop('Id', axis=1, inplace=True)

X = housing.copy()
X.drop('Id', axis=1, inplace=True)

ord_col = [ 'LotShape',
            'LandSlope',
            'ExterQual',
            'ExterCond',
            # 'BsmtQual',
            # 'BsmtCond'
            ]
#Data exploration
###########
# X_num = X.select_dtypes(exclude="object")
# X_num.sort_values(by=['Expensive'],inplace=True)
# X_num.reset_index(inplace=True,drop=True)
# X_num.iloc[:,9:21].plot(subplots=True,marker='.',linestyle='none')


y = X.pop("Expensive")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_test_compet = X_test_compet[X_test.columns]

X_num = X.select_dtypes(exclude="object")
X_cat = X.select_dtypes(include="object")
oh_col = [e for e in list(X_cat.columns) if e not in ord_col]

numeric_pipe = make_pipeline(
    SimpleImputer(strategy="mean"))

ordinal_cols = X_cat.columns.get_indexer(ord_col)
onehot_cols = X_cat.columns.get_indexer(oh_col)
#onehot_cols = X_cat.loc[:,oh_col].columns

cat_list1 = ['Reg', 'IR1', 'IR2', 'IR3', 'N_A']
cat_list2 = ['Gtl', 'Mod', 'Sev', 'N_A']
cat_list3 = ['Ex','Gd','TA','Fa','Po', 'N_A']


categorical_encoder = ColumnTransformer(
    transformers=[
        ("cat_ordinal", 
         OrdinalEncoder(categories=[cat_list1,cat_list2,cat_list3,cat_list3]),
         ordinal_cols),
        
        ("cat_onehot", 
         OneHotEncoder(handle_unknown="ignore"), 
         onehot_cols),
    ]
)

categorical_pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value="N_A"),
                                 categorical_encoder
                                )
full_preprocessing = ColumnTransformer(
    transformers=[
        ("num_pipe", numeric_pipe, X_num.columns),
        ("cat_pipe", categorical_pipe, X_cat.columns),
    ]
)

full_pipeline = make_pipeline(full_preprocessing,
                              LogisticRegression(max_iter = 1000)
                              #DecisionTreeClassifier()
                              )
                   

full_pipeline.fit(X = X_train, y = y_train)

y_pred = full_pipeline.predict(X_test)
y_pred_compet = full_pipeline.predict(X_test_compet)

X_test_compet = pd.read_csv(path+"test.csv")
out = pd.DataFrame(y_pred_compet)
out = out.join(X_test_compet['Id'])
out.rename(columns={0: 'Expensive'},inplace=True)
out = out[['Id','Expensive']]
out.to_csv('Dzmitry_submission01.csv', index=False)

acc = accuracy_score(y_true = y_test, y_pred = y_pred)

# param_grid = {
#     #'my_preproc__num_pipe__num_pipe_imputer__strategy':['mean','median','most_frequent'],
#     'decisiontreeclassifier__max_depth': range(3, 5),
#     #'decisiontreeclassifier__min_samples_leaf': range(5, 10),
#     #'decisiontreeclassifier__min_samples_split': range(2, 16, 5),
#     #'decisiontreeclassifier__criterion':['gini', 'entropy']
#     }

# search = GridSearchCV(full_pipeline, 
#                       param_grid, 
#                       cv=5,
#                       scoring='accuracy',  
#                       verbose=1)

# search.fit(X = X_train, y = y_train)

# print('Best score:')
# print(search.best_score_)
# print('Best parameters:')
# print(search.best_params_)

#y_pred = search.predict(X_test)
#acc = accuracy_score(y_true = y_test, y_pred = y_pred)

print('Accuracy on test:')
print(acc)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')