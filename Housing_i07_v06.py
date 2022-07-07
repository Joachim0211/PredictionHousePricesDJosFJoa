import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars
from my_utilities import DropHighCorrCol, plot_predictions

# get the start time
st = time.time()

path = '..\Data\\'

#housing = pd.read_csv(path+"train_kaggel.csv")
housing = pd.read_csv(path+"housing_prices-iter7.csv")

X = housing.copy()
X.drop('Id', axis=1, inplace=True)
y = X.pop("SalePrice")

X0=X.copy()

X_num = X.select_dtypes(exclude="object")
X_cat = X.select_dtypes(include="object")

ord_col_names = [ 'LotShape',
            'LandSlope',
            'ExterQual',
            'ExterCond',
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinType2',
            'HeatingQC',
            'KitchenQual'
            ]
oh_col_names = [e for e in list(X_cat.columns) if e not in ord_col_names]

X_ord = X[ord_col_names]
X_oh = X[oh_col_names]

cat_list1 = ['Reg', 'IR1', 'IR2', 'IR3', 'N_A']#LotShape
cat_list2 = ['Gtl', 'Mod', 'Sev', 'N_A']#LandSlope
cat_list3 = ['Ex','Gd','TA','Fa','Po', 'NA','N_A']#ExterQual, ExterCond, BsmtQual, BsmtCond,...HeatingQC, KitchenQual
cat_list4 = ['Gd','Av','Mn','No', 'NA','N_A']#BsmtExposure
cat_list5 = ['GLQ','ALQ','BLQ','Rec','LwQ','Unf', 'NA','N_A']#BsmtFinType1, BsmtFinType2

cat_lists_all = [cat_list1, cat_list2, cat_list3,
                 cat_list3, cat_list3, cat_list3,
                 cat_list4, cat_list5, cat_list5,
                 cat_list3, cat_list3]

num_pipe = make_pipeline(SimpleImputer(strategy="mean"),
                         #StandardScaler()
                         #RobustScaler()
                         MinMaxScaler(feature_range=(0,1))
                         )
                         
                         
ord_pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value="N_A"),
                         OrdinalEncoder(categories=cat_lists_all),
                         MinMaxScaler(feature_range=(0,1))
                        )
oneH_pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value="N_A"),
                         OneHotEncoder(handle_unknown="ignore")
                        )

full_preprocessing = ColumnTransformer(
    transformers=[
        ("n_pipe", num_pipe, X_num.columns),
        ("o_pipe", ord_pipe, X_ord.columns),
        ("oh_pipe", oneH_pipe, X_oh.columns),
    ]
)

drop_hcorr_tfm = FunctionTransformer(DropHighCorrCol)

full_pipeline = make_pipeline(full_preprocessing,
                              #drop_hcorr_tfm,
                              #LassoLars()
                              Ridge(),
                              #ElasticNet()
                              #Lasso(),
                              #LinearRegression()
                              )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

full_pipeline.fit(X = X_train, y = y_train)

# z = full_pipeline.fit_transform(X = X_train, y = y_train)
# z=pd.DataFrame(z)
# zd=z.describe()

y_pred = full_pipeline.predict(X_test)

y_test = y_test.reset_index(drop=True)

err = mape(y_true = y_test.reset_index(drop=True), y_pred = y_pred)
print('Error on test (mape):')
print(err)

#plot_predictions(y_test, y_pred)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


#sns.PairGrid(X_num.iloc[:,:3]
