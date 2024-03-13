import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

data = pd.read_csv('winemag-data-130k-v2.csv')
data.head(20)
data.info()
data.describe()
data[['country','variety','winery','taster_name','taster_twitter_handle','province','region_1','region_2']].describe()
data.country.value_counts().head(15).plot.barh(width=0.9,figsize=(10,6),color='darkred');
(data.country.value_counts(normalize=True)*100).head(6)
data.variety.value_counts().head(70)
US = data[data['country'] == 'US']
US.head()
years = US.title.str.extract('([1-2][0-9]{3})').astype(float)
years[years < 1990] = None
US = US.assign(year = years)
US=US.dropna(subset=['price'])
plt.scatter(x=US[US['variety'] == 'Pinot Noir']['points'],y=US[US['variety'] == 'Pinot Noir']['price'],c=US[US['variety'] == 'Pinot Noir']['year']);
US[US['variety'] == 'Pinot Noir']
sns.boxplot(x='variety', y='year', data = US[US['variety'] == 'Pinot Noir'])
sns.jointplot(x='year',y='price',data=US[US['variety'] == 'White Blend']);
US = US.drop_duplicates('description')
US.shape
US.variety.value_counts()
US = US.groupby('variety').filter(lambda x: len(x) >500)
wine_us =US.variety.unique().tolist()
wine_us.sort()
from sklearn.model_selection import train_test_split
X = US.drop(['Unnamed: 0','country','designation','points','province','taster_name',
       'taster_twitter_handle', 'title','region_1','region_2','variety','winery'], axis = 1)
print(X.columns, "X columns")
y = US.variety
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
output = set()
for x in US.variety:
    x = x.lower()
    x = x.split()
    for y in x:
        output.add(y)
variety_list =sorted(output)
extras = ['.', ',', '"', "'", '?', '!', ':', ';','-' ,'(', ')', '[', ']', '{', '}', 'cab',"%"]

from nltk.corpus import stopwords
import nltk
stop = set(stopwords.words('english'))
stop.update(variety_list)
stop.update(extras)
US.variety.value_counts()
US.head()
sum(US['year']>2002)
data_=US[US['year']>2002]
plt.figure(figsize=(10,4))
data_.variety.value_counts().plot.bar()
rep_v = data_.groupby('variety')['year'].agg('median')
for i in rep_v.index :
    data_.loc[(data_['year'].isna()) & (data_['variety'] == i),'year']=rep_v[i]
data_.variety.describe()
X_ = data_.drop(['Unnamed: 0','designation','country','taster_name',
       'taster_twitter_handle', 'title','region_1','region_2','variety','winery'], axis = 1)
y = data_.variety
print("Length of y:", len(y))
print("Length of X:", len(X_))
X = pd.get_dummies(X_,columns=["province"])
X_apriori = X.copy()
# drop all features except one hot encoded province
columns_to_drop = [
    'description', 'price', 'year', 'points'
]
for col in columns_to_drop:
    if col in X_apriori.columns:
        X_apriori.drop(col, axis=1, inplace=True)
province_columns = [col for col in X.columns if 'province_' in col]
X[province_columns] = X[province_columns].astype(int)
# X = pd.get_dummies(X_, columns=["province"], drop_first=True)
# X = X.astype(int)
# y = data_.variety
print(y)
X.info()
X.head()


##################PCA#####################
# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import  hstack
# Copy the dataset
df_for_PCA = X.copy()
# stop_list = list(stop)
# vect = CountVectorizer(stop_words=stop_list)
# X_dtmm = vect.fit_transform(df_for_PCA.description)
# Xss = df_for_PCA.drop(['description'], axis=1).to_numpy()
# X_combined = hstack((Xss, X_dtmm)).todense()
# X_combined_dense = np.asarray(X_combined)

# Standardize the data
scaler = StandardScaler()
# show columns in df_for_PCA
print(df_for_PCA.columns, "df_for_PCA.columns")
# drop description column for now and all province_ columns
df_for_PCA = df_for_PCA.drop(['description'], axis=1)
columns_to_drop = [
    'province_America', 'province_Arizona', 'province_California',
    'province_Colorado', 'province_Connecticut', 'province_Idaho',
    'province_Kentucky', 'province_Massachusetts', 'province_Michigan',
    'province_Missouri', 'province_Nevada', 'province_New Jersey',
    'province_New Mexico', 'province_New York', 'province_North Carolina',
    'province_Ohio', 'province_Oregon', 'province_Pennsylvania',
    'province_Texas', 'province_Virginia', 'province_Washington',
    'province_Washington-Oregon'
]

# Dropping columns from df_for_PCA
for col in columns_to_drop:
    if col in df_for_PCA.columns:
        df_for_PCA.drop(col, axis=1, inplace=True)
# print df_for_PCA.columns
print(df_for_PCA.columns, "df_for_PCA.columns")
# make a for loop to drop all province_ columns

X_std = scaler.fit_transform(df_for_PCA)






# Drop the specified columns
# display df_for_PCA

#one-hot encode the day of week column
# df_for_PCA = pd.get_dummies(df_for_PCA, columns=["day_of_week"])

# Standardize the data to have a mean of ~0 and a variance of 1

# Create a PCA instance: pca
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_std)

# Plot an elbow curve to find the optimal number of components

elbow = PCA().fit(X_std)
plt.plot(np.cumsum(elbow.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

print("The first two principal components explain {}% of the variance.".format(
    round(sum(elbow.explained_variance_ratio_[:2])*100, 2)))

print(elbow.explained_variance_ratio_,"elbow.explained_variance_ratio_")
# plot elbow plot for above elbow variable
plt.plot(elbow.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


df_for_PCA.head()

##################PCA#####################


########heat map################
# show the heatmap for df_for_PCA
sns.heatmap(df_for_PCA.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Coefficient')
plt.show()

# show the heatmap for df_for_PCA using pearson correlation
sns.heatmap(df_for_PCA.corr(method='pearson'), annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Coefficient')
plt.show()


# do t test for df_for_PCA
# add variety to df_for_PCA from data
from scipy.stats import ttest_ind
df_for_PCA['variety'] = data_.variety
numerical_columns = ["price", "year", "points"]
print(df_for_PCA.head(), "check this")
df_for_PCA['variety_numerical'] = pd.Categorical(df_for_PCA['variety']).codes
df_for_PCA = df_for_PCA.drop('variety', axis=1)
df_for_PCA_for_f_test = df_for_PCA.copy()
def t_test(data, target, columns):
    for col in columns:
        t_value, p_value = ttest_ind(data[data[target] == 0][col],
                                     data[data[target] == 1][col],
                                     nan_policy='omit')  # Omit NaN values
        print(f'T-value for {col} vs {target}: {t_value}')
        print(f'P-value for {col} vs {target}: {p_value}')

# Perform the t-test
t_test(df_for_PCA, 'variety_numerical', numerical_columns)

#perform the f test
from scipy.stats import f_oneway
def f_test(data, target, columns):
    unique_varieties = data[target].unique()
    groups = [data[data[target] == variety][col] for variety in unique_varieties for col in columns]
    f_value, p_value = f_oneway(*groups)
    print(f'F-value for {col} vs {target}: {f_value}')
    print(f'P-value for {col} vs {target}: {p_value}')

# Perform the F-test
f_test(df_for_PCA, 'variety_numerical', numerical_columns)

########After t test################

new_df = df_for_PCA.copy()

X_train, X_test, y_train, y_test = train_test_split(
    new_df.drop('variety_numerical', axis=1),
    new_df['variety_numerical'],
    test_size=0.3,
    random_state=5805  # Set a random seed for reproducibility
)

import statsmodels.api as sm

model = sm.OLS(y_train, sm.add_constant(X_train)).fit()  # Add a constant term
print(model.summary())

f_statistic = model.fvalue
p_value = model.f_pvalue

print('F statistic = ' + str(f_statistic))
print('P-value = ' + str(p_value))

# plot train, test, and predicted values in one plot
plt.scatter(X_train['points'], y_train, color='blue', label='train')
plt.scatter(X_test['points'], y_test, color='red', label='test')
plt.scatter(X_test['points'], model.predict(sm.add_constant(X_test)), color='green', label='predicted')
plt.xlabel('points')
plt.ylabel('variety')
plt.legend()
plt.show()

# Confidence interval Analysis
import statsmodels.stats.api as sms

confidence_interval = model.conf_int()
print(f"Confidence interval:\n{confidence_interval}")


final_confidence_interval = model
y_pred = model.predict(sm.add_constant(X_test))

#Calculate  and print R squared, adjusted R - square, AIC, BIC and MSE

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from math import sqrt

r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
aic = model.aic
bic = model.bic
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adj_r_squared}")
print(f"AIC: {aic}")
print(f"BIC: {bic}")
print(f"MSE: {mse}")









from sklearn.feature_extraction.text import CountVectorizer
#generando variable de textmining

from scipy.sparse import csr_matrix, hstack
stop_list = list(stop)
vect = CountVectorizer(stop_words = stop_list)
X_dtm = vect.fit_transform(X.description)
Xs = X.drop(['description'],axis=1).to_numpy()
print(Xs)
X_dtm = hstack((Xs,X_dtm))
X_dtm_dense = X_dtm.todense()
df_X_dtm = pd.DataFrame(X_dtm_dense)

# View the DataFrame
print(df_X_dtm.head())
print(X_dtm, "X_dtm")
len(y)
print("Length of X_dtm:", X_dtm.shape[0])
print("Length of y:", len(y))
Xtrain,Xtest,ytrain,ytest = train_test_split(X_dtm,y,random_state=1)





##############Phase 3#####################
Xtrain,Xtest,ytrain,ytest = train_test_split(X_dtm,y,random_state=1)
print("new Length of y:", len(y))
wine=y.unique()
from sklearn.linear_model import LogisticRegression
models = {}
for z in wine:
    model = LogisticRegression()
    y_binary = ytrain == z
    model.fit(Xtrain, y_binary)
    models[z] = model
testing_probs = pd.DataFrame(columns = wine)
len(wine)
probs = pd.DataFrame(columns = wine)
for z in wine:
    probs[z] = models[z].predict_proba(Xtest)[:,1]
probs.head()
probs_=probs.fillna(0)
pred = probs.idxmax(axis=1)
comparison = pd.DataFrame({'actual':ytest.values, 'predicted':pred.values})
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")\

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report, recall_score, \
    confusion_matrix, mean_squared_error, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsRegressor
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.neural_network import MLPClassifier
x_train, x_test, y_train, y_test = train_test_split(X_dtm, y, test_size=0.2, random_state=5805)

# CREATING ALL THE MODELS

models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('SVM', SVC(probability=True)))
models.append(('MLP', MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)))
models.append(('GaussianNB', GaussianNB()))

# RUNNING OF THE MODELS
# IN PY CHARM CHECK THE PLOT FOR CONFUSION MATRIX

for name, model in models:
    print()
    print()
    # Train model
    if name in ['GaussianNB', 'MLP']:
        model.fit(x_train.toarray(), y_train)
        predictions = model.predict(x_test.toarray())
    else:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    # Print accuracy
    print(name)
    print(f"Training Accuracy: {accuracy_score(y_train, model.predict(x_train.toarray() if name == 'GaussianNB' else x_train)):.3f}")
    print(f"Testing Accuracy: {accuracy_score(y_test, predictions):.3f}\n")

    # Classification report
    print(classification_report(y_test, predictions, zero_division=0))

    # Plot ROC Curve for each class
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    n_classes = y_test_bin.shape[1]
    y_pred_proba = model.predict_proba(x_test.toarray() if name == 'GaussianNB' else x_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve by class')
    plt.legend(loc="lower right")
    plt.show()


    # Plotting Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y))
    # dont plot consfusion matrix but display it
    print(cm,"Confusion matrix")



################PHASE 4#####################
# # K-mean clustering with silhouette score for the k-selection within cluster variation plot
#
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
#
# # Create empty lists to store the scores for the evaluation metrics
silhouette_scores = []
inertia_scores = []

# Create a list of possible k values
k_values = list(range(2, 11))

# For each k value
for k in k_values:
    # Create a KMeans instance with k clusters
    kmeans = KMeans(n_clusters=k, random_state=5805)
    # Fit the model to the data
    kmeans.fit(X_dtm)
    # Append the average silhouette score and inertia score to the respective lists
    silhouette_scores.append(silhouette_score(X_dtm, kmeans.labels_))
    inertia_scores.append(kmeans.inertia_)

# Plot the silhouette score vs k values
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()

# Plot the inertia score vs k values
plt.plot(k_values, inertia_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia Score')
plt.title('Inertia Score vs Number of Clusters')
plt.show()


# phase 4 part 2
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
#
# # Create an empty list to store the silhouette scores
# silhouette_scores = []
#
# # Define a range of possible eps values
# eps_values = [0.001, 0.01, 0.1, 1, 10, 100]
#
# # Iterate over different eps values
# for eps in eps_values:
#     # Create a DBSCAN instance with the current eps value
#     dbscan = DBSCAN(eps=eps)
#     # Fit the model to the data
#     dbscan.fit(X_dtm)
#
#     # Check if more than one cluster has been formed (excluding noise points)
#     if len(set(dbscan.labels_)) > 1:
#         # Calculate silhouette score and append it to the list
#         silhouette_scores.append(silhouette_score(X_dtm, dbscan.labels_))
#     else:
#         # Append a default value (e.g., -1) if only one cluster is formed
#         silhouette_scores.append(-1)
#
# # Plot the silhouette scores against the eps values
# plt.plot(eps_values, silhouette_scores)
# plt.xlabel('Eps values')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs Eps values')
# plt.show()
#

# from sklearn.cluster import DBSCAN
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import numpy as np
# import pandas as pd
#
# # Assuming X_dtm is already defined and is your dataset
#
# # Create a pipeline that standardizes the data and then applies PCA
# pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95))
#
# # Fit the pipeline and transform the data
# X_dtm_pca = pca_pipeline.fit_transform(X_dtm)
#
# # Sample a subset of your data if it's too large
# # For example, sample 10% of your data
# sample_idx = np.random.choice(X_dtm_pca.shape[0], int(X_dtm_pca.shape[0] * 0.1), replace=False)
# X_dtm_sample = X_dtm_pca[sample_idx, :]
#
# # Define DBSCAN with a larger eps and min_samples
# dbscan = DBSCAN(eps=5, min_samples=10)
#
# # Fit DBSCAN to the sample data
# dbscan.fit(X_dtm_sample)
#

from mlxtend.frequent_patterns import apriori, association_rules

# Assuming 'df_for_PCA' is preprocessed and contains only one-hot encoded features
# Apply the Apriori algorithm to find frequent itemsets
# select
frequent_itemsets = apriori(X_apriori, min_support=0.01, use_colnames=True)

# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Display the top association rules sorted by confidence
rules = rules.sort_values(by='confidence', ascending=False)
print(rules.head(), "rules.head()")




