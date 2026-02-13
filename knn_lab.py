# %% [markdown]
# # KNN Lab II

# %% [markdown]
# Building a knn model based on the College Completion Dataset

# %% [markdown]
# ## Data Cleaning

# %%
# Importing Packages
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data
from sklearn.neighbors import KNeighborsClassifier


# %%
# The dataset is very convoluted, so let's clean it up a bit. 

colleges = pd.read_csv("data/cc_institution_details.csv")
colleges.info()


# %%
# This code is framed from the lab instructions:

to_drop = list(range(40, 57)) # all the vsa columns
to_drop.extend([0,1,2,3,4,7,10,11,12,18,19,20,23,24,25,27,28,29,31,33,35,37,57,60,61,])

colleges_drop = colleges.drop(colleges.columns[to_drop], axis=1)

# Fixing HBCU and flagship columns so it's usable (replace with 0s and 1s):
colleges_drop['hbcu'] = [1 if colleges_drop['hbcu'][i]=='X' else 0 for i in range(len(colleges_drop['hbcu']))]
colleges_drop['flagship'] = [1 if colleges_drop['flagship'][i]=='X' else 0 for i in range(len(colleges_drop['flagship']))]

# Since kNN can't work with missing values, we need to drop them:
colleges_nomiss = colleges_drop.dropna()

# Changing categorical columns to the "category" data type:
cols = ["level", "control", "hbcu","flagship"]
colleges_nomiss[cols] = colleges_nomiss[cols].astype('category')

# One-Hot Encoding
category_list = list(colleges_nomiss.select_dtypes('category'))   # list of all category vars
colleges_encoded = pd.get_dummies(colleges_nomiss, columns=category_list)
colleges_encoded = colleges_encoded.astype(int)

# Let's also drop hbcu_0 and flagship_0 since it's redundant information:
colleges_clean = colleges_encoded.drop(columns=["hbcu_0", "flagship_0"], errors="ignore")

colleges_clean.info()

# Now we can work with the data


# %% [markdown]
# ## 1. Target Variable and Question

# %%
# So it turns out I somehow misinterpreted the column "grad_150_value" from 
# the previous assignment. The Kaggle page of the data was incredibly confusing
# to me and they listed a column called "grad_150" that listed the actual number
# of students the graduated within 6 years instead of the percentage. So let's 
# clear everything up here:
#
# Question: How well can college characteristics predict whether a college has a 
# 6-year graduation rate above 60%?
#
# Target Variable: Graduation rate within 6 years above 60% ("high_completion")

colleges_clean['high_completion'] = (colleges_clean['grad_150_value'] >= 60).astype(int)


# %%
# Here's the prevelance of the target variable:

prevalence = colleges_clean['high_completion'].mean()
print(prevalence)

# Looks like about 25.9% of colleges have 6 year grad rate of 60% or higher. 


# %% [markdown]
# ## Step 2: KNN Model



# %% 
# Separating Training and Test Data:

# Separate Target Variable
y = colleges_clean["high_completion"]
X = colleges_clean.drop(columns=["high_completion","grad_150_value"])

# Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=83,
    stratify=y     
)

# %%
# Standardizing the Data


# Numeric predictors only
numeric_cols = X_train.select_dtypes(include="number").columns

# Defining Scaler
scaler = MinMaxScaler()

# Scaling Data
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# We use .fit_transform for X_train because it finds the min and max for each
# column (fitting) and then scales the values so they're in between 0 and 1 
# (transforming). We only use .transform for X_test because it uses the same
# minimums and maximums from the training data. We want to ensure both the 
# training and test sets are scaled the same way. 

X_train.describe()

# %%
# Compare with test set 
X_test.describe()


# %%
# Creating the model with 3 nearest neighbors

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


# %% [markdown]
# ## 3. Dataframe: Target values, Predicted values, and Test Probabilities 


# %%
# Actual Target Values
y_actual = y_test.values

# Predicted Values:
y_hat = model.predict(X_test)

# Probability of Positive Class (6-year grad rate > 60%)
y_prob = model.predict_proba(X_test)[:, 1]

# Combine into dataframe:
results = pd.DataFrame({
    "actual": y_actual,
    "predicted": y_hat,
    "prob_positive": y_prob
})

results.head()

# %%
# Confusion Matrix
pd.crosstab(y_actual, y_hat, rownames=["Actual"], colnames=["Predicted"])

# %% [markdown]
# ## 4. No code question
# If you adjusted the k hyperparameter what do you think would happen to 
# the threshold function? Would the confusion matrix look the same at the same
# threshold levels or not? Why or why not?

# %% [markdown]
# The threshold function describes how the knn model converts the predicted probabilities
# of the positive class (prob_positive above) into definitive classifications (1 or 0).
# As an example, if k=3 like it currently is, and if 2 neighbors are of the positive class,
# then the threshold function would convert the predicted probability of 0.667 (2/3) into
# a positive classification of 1 since 0.667 > 0.5. 
#
# If I adjusted k to a different value, then the predicted probabilities would change because
# there would be a different amount of nearest neighbors to take into account. However, 
# the threshold function itself wouldn't change since the threshold or deciding a class as
# 0 (6 year grad rate below 60%) or 1 (above 60%) remains at a probability of 0.5. Instead,
# the confusion matrix would look different at different k values even if the threshold values
# are the same. This is because different k values means each observation would have a different
# set of neighbors, leading to different probability estimates. For example, if k=3, the 
# probability of a data point being in the positive class could be 2/3, making it be predicted to
# be of class 1. But if k=5, then the next 2 nearest neighbors might be of class 0 and bring the
# probability down to 2/5, where the model would classify it as class 0 since the probability 
# is less than 0.5. Therefore, some observations may be above or below the threshold depending
# on what the k value is, meaning the confusion matrix would not look the same at the same 
# threshold levels.


# %% [markdown]
# ## 5. Evaluating Confusion Matrix
#
# Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
# concerns or positive elements do you have about the model as it relates to your question? 


# %%
# Looking at the numbers:
#
# True Negatives (predicted 0 and is 0): 470
# False Negatives (predicted 0 and got 1): 36
# True Positives (predicted 1 and got 1): 139
# False Positives (predicted 1 and got 0): 30

# Accuracy
accuracy = (470+139)/(470+139+36+30)
print("Accuracy: " + str(accuracy))

# Sensitivity
# 
# The sensitivity is how often the model correctly predicts a college with a 6-year
# graduation rate above 60%.
sensitivity = 139/(139+36)
print("Sensitivity: " + str(sensitivity))

# Specificity
#
# The specificity is how often the model correctly predicts a college with a 6-year
# graduation rate below 60%.
specificity = 470/(470+30)
print("Specificity: " + str(specificity))


# %% [markdown]
# Connection to Question:
#
# My question was: how well can college characteristics predict whether a college
# has a 6-year graduation rate above 60%? Looking at the confusion matrix, it seems 
# that the model performs well overall with an accuracy of 90.7%. One positive element
# I have on the model is that it does a great job of predicting colleges that do not have
# a 6-year graduation rate above 60% as the model has a specificity of 94.4%. However, 
# one element of the model I'm concerned about is how it predicts colleges with a 6-year
# graduation rate above 60%. The sensitivity of the model is around 80%, which means the model
# correctly postively classifies a college 80% of the time. I'm not completely certain of
# what a "good" sensitivity of a model is, but I'm still a little concerned about my model
# missing these high-completion colleges. However, it's important to remember the prevelance 
# of the positive class, which is around 26%, meaning that high-completion colleges are less
# common and therefore, harder to predict since detecting something that happens more 
# infrequently is harder. Overall, my model does a reasonably good job of answering my question.


# %% [markdown]
# ## 6. Functions
#
# Create two functions: One that cleans the data & splits into training|test and one that 
# allows you to train and test the model with different k and threshold values, then use them to 
# optimize your model (test your model with several k and threshold combinations). Try not to use 
# variable names in the functions, but if you need to that's fine. (If you can't get the k function 
# and threshold function to work in one function just run them separately.)


# %%
# Function 1: Cleaning & Splitting Data


def data_prep(df, target_col, test_size=0.2, random_state=83):

    # Copy so no mess w/ original data
    data = df.copy()

    # Defining target variable
    y = data[target_col]
    X = data.drop(columns=target_col)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale Numeric Predictors
    numeric_cols = X_train.select_dtypes(include="number").columns

    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test



# %%
# Function 2: Train/Test w/ k and Threshold Values


def knn_model(X_train, X_test, y_train, y_test, k=3, threshold=0.5):

    # Creating and fitting model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Probability of positive class
    probs = model.predict_proba(X_test)[:, 1]

    # Mapping probabilities to predictions
    preds = (probs >= threshold).astype(int)

    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
        "prob_positive": probs
    })

    cm = pd.crosstab(results["actual"], results["predicted"])

    return results, cm


# %%
# Let's add a third function to evaluate the model:

def evaluate_knn(X_train, X_test, y_train, y_test, k=3, threshold=0.5):

    # Using Previous Function to get the results dataframe & confusion matrix
    results, cm = knn_model(X_train, X_test, y_train, y_test, k, threshold)

    tn = cm.loc[0, 0]   # true negatives
    fp = cm.loc[0, 1]
    fn = cm.loc[1, 0]
    tp = cm.loc[1, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "k": k,
        "threshold": threshold,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    }





# %%
# Optimizing Model

# Getting training/testing data
X_train, X_test, y_train, y_test = data_prep(colleges_clean, 'high_completion')

# Finding Best k and threshold w/ a table:

results_list = []

for k in [3, 5, 7, 9, 11, 13]:   # diff k values
    for threshold in [0.3,0.4, 0.5, 0.6, 0.7]:   # diff threshold values
        # Evaluating Model
        metrics = evaluate_knn(               
            X_train, X_test, y_train, y_test,      
            k=k,
            threshold=threshold
        )
        results_list.append(metrics)

# Making Dataframe
results_df = pd.DataFrame(results_list)
results_df




# %%
# Sort dataframe by accuracy:

results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

# %%
# Sort by sensitivity (predicting high-completion colleges):

results_df.sort_values("sensitivity", ascending=False).reset_index(drop=True)

# %%
# Sort by specificity (predicting low-completion colleges):

results_df.sort_values("specificity", ascending=False).reset_index(drop=True)

# %% [markdown]

# After testing my model with several different k and threshold values, there isn't just
# one true "optimized" model. Soley based on accuracy, the model is best at a k of 7 and a
# threshold of 0.6. However, looking at sensitivity, which is better suited to our question,
# then the best k is 3 with a threshold value of 0.3 (although the specificity is the lowest
# compared to the others). 


# %% [markdown]
# ## 7. Reflection

# How well does the model perform? Did the interaction of the adjusted thresholds 
# and k values help the model? Why or why not?

# Overall, the model performed decently well. Across a variety of threshold and k values,
# the model almost always obtains an accuracy of above 90%, which is solid. However,
# the sensitivity across these ranges varies much more, with some being as low as 62.2%
# and others being as a high as 96%. This is significant because our question was how 
# well we could use colleges characteristics to predict colleges with a 6-year graduation
# rate above 60%, and the prevelance directly corresponds to this question. The specificity
# (how well the model predicted low-completion colleges) is generally on the higher side 
# hovering around the 90% range and above, but this is expected since there are more 
# low-completion colleges than high ones in the dataset. 
#
# The interaction of adjusted thresholds and k values defintely helped the model. Initially 
# when we used k = 3 and threshold = 0.5, our model did an ok job with a 90% accuracy but
# not a very high prevelance (80%). Testing the model with different k and threshold values
# did a great job improving our metrics, as we now see values that exceed each of the initial
# values for accuracy, sensitivity, and specificity when k = 3 and the threshold = 0.5,
# making us better suited to answer our initial question.


# %% [markdown]
# ## 8. New Model for New Variable


# %%
# New Target Variable: HBCU status
#
# For reference, here's the prevelance:
prev = colleges_clean["hbcu_1"].mean()  # hbcu_1 is from the one-hot encoding from before
print(prev)

# Much lower prevelance that last time: 2.73%.

# %%
# First let's get the training and testings sets from our first function:

X_train, X_test, y_train, y_test = data_prep(colleges_clean, target_col="hbcu_1")


# %%
# Now let's actually create the model using the second function:

results, confusion_matrix = knn_model(
    X_train, X_test,
    y_train, y_test,
    k=3,
    threshold=0.5
)

print(confusion_matrix)

# Looks the model is really good at predicting a college to not be an HBCU but not
# the best at predicting colleges that actually are HBCUs.


# %%
# Now we evaluate:

results_list = []

for k in [3, 5, 7, 9, 11, 13]:
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = evaluate_knn(
            X_train, X_test,
            y_train, y_test,
            k=k,
            threshold=threshold
        )
        results_list.append(metrics)

results_df = pd.DataFrame(results_list)
results_df.sort_values("accuracy", ascending=False).reset_index(drop=True)


# %%
# Looking at the results might be a little misleading because of the higher accuracies.
# We have to keep in mind that the prevelance of an HBCU school is less than 3%, meaning
# that a human should be able to predict whether or not a college is an HBCU school about
# 97% of the time. The important metric to look at is sensitivity, since that's how well
# the model is able to predict colleges that are truly HBCUs:

results_df.sort_values("sensitivity", ascending=False).reset_index(drop=True)

# The highest sensitivity occurs at a k value of 3 and a threshold value of 0.3, which
# is the same thing for the graduation rate model, which makes sense since a lower threshold
# and a lower k increases the chances that the model will group a data point in the positive
# class. Still, this model achieved a sensitivity of only 77.78%, but it also had the lowest
# accuracy out of all the parameter variations at 96.89%, which is signficant since it's actually
# less than 97%. In order to improve this model, either more data will be needed to train it or
# somehow new features could be added that are better predictive of HBCU status. 