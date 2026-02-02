# %% [markdown]
# # Machine Learning Bootcamp Lab

# %%
# Importing Packages
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data


# %% [markdown]
# ## College Completion Dataset

# %% [markdown]
# ### Step 1: Reviewing Dataset
#
# This data set contains information about the success and progress 
# of U.S. college students by examing 3793 colleges. Each row in the 
# dataset is a college, and there are 63 total columns. The columns 
# contain information about the college location, type, awards, and transfers.

# %% 
# Reading in the dataset:
#
# Source: https://www.kaggle.com/datasets/thedevastator/boost-student-success-with-college-completion-da/data
colleges = pd.read_csv("data/cc_institution_details.csv")

# Analyzing the info of the dataset:
colleges.info()


# %% [markdown]
# #### Potential Problems
# 
# %% [markdown]
# Note: I think I misinterpreted this part and went in depth about what
# might be wrong with the dataset itself.
#
# Looking at colleges.info() and Data Wrangler, here are some 
# problems that could mess up the analysis:
#
# - Multiple columns have a significant amount of missing values (shown below),
#   including hbcu, flagship, med_sat_value, med_sat_percentile, endow_value, 
#   vsa_year, nicknames, and all the columns starting with "vsa."
#
# %%
# Generating columns with at least 50% of values missing
#
# fraction of missing values per column
missing_fraction = colleges.isnull().mean()  

# Filter columns with at least 50% missing
high_missing = missing_fraction[missing_fraction >= 0.5]

# Sort descending
high_missing_sorted = high_missing.sort_values(ascending=False)

print(high_missing_sorted)  # that's a lot of columns

# %% [markdown]
# - There are many columns with object type string (shown below), such as 
#   the college name, city, state, nickname, and even whether or not it's 
#   an HBCU (which would probably be better as a boolean). We would want 
#   to change these to category variables.

string_cols = list(colleges.select_dtypes(include='str').columns) # selecting string dtype
print(string_cols)  

# %% [markdown]
# - The column of cohort size has a minimum value of 1 and a maximum value 
#   of 16229, meaning that extreme values might be present and we might have
#   to normalize/scale the data. Other columns have a similar problem, such as 
#   endow_value and fte_value.
#
# - Some columns might not be completely useful, such as the index column.

# %% [markdown]
# What might the dataset address?
#
# - Predicting graduation rates or transfer rates
# - How features of colleges differ across states
# - The differences in public and private colleges
# - How awards compare to big versus small schools


# %% [markdown]
# #### Some Questions
#
# - What states have the most colleges in the dataset?
# - Which colleges get the most awards and spend the most on them?
# - Which colleges have a lot of transfers who graduate elsewhere?
# - How do rural colleges compare and contrast with urban colleges?
# - Are there any geographic patterns in getting awards and spending?


# %% [markdown]
# ### Step 2: Data Preparation

# %% [markdown]
# #### Generic Question Dataset can Address:
#
# What factors are associated with higher graduation rates in colleges?
# 
#
# #### Target Variable:
# Graduation rates of colleges (making it a regression problem)
#
# #### Independent Business Metric:
# With new cohorts coming in each year, how can the business (the college) 
# use a model based on the findings from the data to evaluate the accuracy?
# An independent business metric could be the change in the graduation
# rates over time. It can also even be the change in profit 
# over time, since ultimately, college enrollment plays a big role in revenue.
# Colleges can make policy changes based on a model's predictions about what
# affects graduation rates, and they can see how these numbers change
# over time to see if these policies work. 

# %%
# #### Dropping Unneeded Variables

# %%
# There's a few columns that aren't completely relevant, so let's drop them
# so that we can simplify an already cluttered data frame. 
#
# Identifier columns we don't really need: index, unitid, site, nickname
# Also probably don't need longitude & latitude since there's already city & state
# The "similar" column also looks confusing, so let's drop that too

columns_to_drop = ['index', 'unitid','site','nicknames',
                   'long_x','lat_y','similar']
colleges = colleges.drop(columns=columns_to_drop)

# Verifying:
colleges.info()
# We see that we're now at 56 columns and not 63 anymore


# %% [markdown]
# #### Creating Target Variable


# %%
# Since the target variable is graduation rate, we need to add that 
# columns to the dataset.
#
# Graduation Rate:
# The dataset has columns for grad_100_value and grad_150_value, which gives 
# the number of students who graduated 100% or 150% of normal time.
# For the target variable, I'll go with graduation rate of 150% of normal time
# since it's more lenient for student who took longer to finish their degree.
# To get a grad rate we simply divide grad_150_value by the cohort size:

colleges['grad_rate'] = (
    colleges['grad_150_value'] /
    colleges['cohort_size']
)

# Verify:
colleges.info()



# %% [markdown]
# #### Prevelance of Target Variable

# %%
# Since this is a regression problem, we calculate the prevelance of the 
# target variable by finding the mean.

colleges['grad_rate'].describe()

# The mean for grad rate is 0.821 (82.1 % graduation rate). However, when looking
# closer at this column, some of the values are above 1, which means there's either 
# errors in the data or some of the students in grad_rate are not included in the 
# original cohort. 

# %%
# One of the rows has a grad_rate of 100 (it should be between 0 and 1),
#  which completely skews the data, so let's drop that row:

colleges = colleges.drop(index=208)

# Let's try the mean again:
colleges['grad_rate'].describe()

# Now it's really low; I'm not liking this dataset at all.

# %%
# We can also see the percentage of colleges with a high graduation rate (say 90%)

percent_above_90 = (colleges['grad_rate'] > 0.9).mean()
print(percent_above_90)

# Looks like only 0.2% of colleges in the dataset have a grad rate above 90%.

# %% [markdown]
# #### Correcting Variable Types

# %%
# Changing categorical columns to the "category" data type:
cols = ["state", "level", "control"]
colleges[cols] = colleges[cols].astype('category')

# Other columns could be changed as well here like hbcu and flagship, 
# but they have so many missing values they likely aren't going to be 
# super relevant.

# Checking to see if the data type changed:
colleges.dtypes.head(10)

# %% [markdown]
# #### Collapsing Factor Levels

# %%
# It might be a little hard to choose a column to factor, but we could do it
# to the state column by keeping the most prevelant states and designated the 
# rest to "other."
#
colleges['state'].value_counts()

# %%
# Let's keep the top 5 states (CA, NY, PA, TX, OH) and reduce the rest to "Other"
#
top_states = ["California","New York","Pennsylvania","Texas","Ohio"]

# Using a lambda function
# If x is in the list above, then change it to type "category", else "Other":
colleges.state = (colleges.state.apply(lambda x: x if x in top_states
                               else "Other")).astype('category')
# Verifying 
colleges.state.value_counts()

# We see that the "Other" dominates the value counts, so it may have been best
# to keep all the states if we wish to perform analysis on how the state column
# affects our target variable.


# %% [markdown]
# #### One-Hot Encoding

# %%
# One-hot encoding is where we convert categorical variables to binary
# indicator columns.
# 
# Right now we have 3 categorical variables, but one-hot encoding the state
# columns doesn't seem very relevant. So that leaves level (4-year or 
# 2-year) and control (public, private for-profit, or private not-for-profit).
# 
category_list = list(colleges.select_dtypes('category'))   # list of all category vars
del category_list[0]   # remove State from the list

# creating new binary columns for each category (level and control)
colleges_encoded = pd.get_dummies(colleges, columns=category_list)
colleges_encoded.columns[-5:]  # checking last 5 column names to verify




# %% [markdown]
# #### Normalizing Continuous Variables

# %%
# First we select all the numeric columns
numeric_cols = list(colleges_encoded.select_dtypes('number'))

# Applying MinMax Scaler:
colleges_encoded[numeric_cols] = MinMaxScaler().fit_transform(colleges_encoded[numeric_cols])

# Verifying --> the min for each numeric column is 0 and the max is 1
colleges_encoded[numeric_cols].describe()



# %% [markdown]
# #### Data Partitions

# %%
# Important Note:
# I'm keeping the target variable of grad_rate in the partitioning,
# but if I was hypothetically going to create a regression model after this,
# I would drop grad_rate when creating the input (x) and output (y) variables.

# For example (code commented out):
# x_train = train.drop(columns=['grad_rate'])
# y_train = train['grad_rate']

# This separates the target variable as from the rest of the data so the model
# can build on how to predict it.

# %%
# First split - Separating training data from the rest
# 55% of data will go to training

train, test = train_test_split(
    colleges_encoded,
    train_size=0.55,
    random_state=42    # random_states fixed the random number generator so code is reproducible
)

# Looking at shapes:
print(f"Training set shape: {train.shape}")
print(f"Remaining set shape: {test.shape}")


# %%
# Second split - Split remaining data into tuning and test sets (50/50)

tune, test = train_test_split(
    test,
    train_size=0.5,
    random_state=42
)

# Looking at shapes:
print(f"Tuning set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")


# %%
# Now we can look at the stats of the target variabl of each partition
# using .describe().

print("Training set target stats:")
print(train['grad_rate'].describe())

print("\nTuning set target stats:")
print(tune['grad_rate'].describe())

print("\nTest set target stats:")
print(test['grad_rate'].describe())

# Overall, they're similar, especially for the train and tune datasets, 
# but again the values for grad_rate vary all over the place.





# %% [markdown]
# ### Step 3: Reflection

# %% [markdown]
# After going through an initial review of the data, my instincts tell me that
# I can partially address the question of factors associated with higher graduation
# rates. To be honest, working with this dataset was a real pain. For one, the actual 
# Kaggle page where this data comes from doesn't include information about all the columns,
# meaning that it was difficult to interpret what all the columns show. Also, the dataset
# is pretty big with over 60 columns to begin with, so strifing through all that information
# was a little overwhelming. 
#
#
# Despite being a large source of data, I still have many concerns if I were to 
# create a regression model to find features associated with high grad rates:

# - Inconsistent Values: 
# Unless I completely messed up or misterpreted the data,
# some of the graduation rates exceed 100%. This is likely due to the inclusion of 
# transfers (making the graduation class greater than the initial cohort) or just general
# error. Without spending a lot of time cleaning up the data to solve this issue, these
# values could otherwise skew the regression analysis greatly.
#
# - Missing Data: 
# Many columns have missing data, with some columns being almost entirely
# composed of missing values. Having so many missing values reduces the usability of 
# observations in the dataset, which in turn may lead to bias in the analysis. 
#
#
# - Overall Data Quality & Feature Relevance: 
# Again, there are so many columns in the dataset, meaning that a significant number of
# columns are likely irrelevant to the prediction of high graduation rates, which introduces
# unncessary cluttering. Also, I'm skeptical on the overall data quality. It's not necessarily
# the dataset's fault, but some numeric columns (specifically grad_value_150) might include
# transfers from other colleges. The fact that this information was not clearly given to us just 
# makes me more uncomfortable when using this data. I just feel like I have too many questions
# and unknowns about the context and reliability of the dataset. 


# %% [markdown]
# Overall, I think the college completion dataset has the ability to answer my 
# question about the factors affecting graduation rates, but it would take a lot of 
# time and effort to clean the data in order to create a valid regression model to
# do so. 






# %% [markdown]
# ## Job Placement Data


# %% [markdown]
# ### Step 1: Reviewing Dataset
#
# This data set contains information about placement information for students at a college
# campus. There are 215 rows and 15 columns (much less than the college completion dataset),
# with details about higher secondary school percentage, degree specialization/field, employability
# test percentage, and salary offers to the placed students. 


# %% [markdown]
# Since the dataset is smaller, here are all the columns and their descriptions:
#
# - sl_no: serial number 
# - gender: M or F 
# - ssc_p: secondary education percentage - 10th grade 
# - ssc_b: Board of Education- Central/ Others 
# - hsc_p: Higher Secondary Education percentage- 12th Grade 
# - hsc_b: Board of Education- Central/ Others 
# - hsc_c: Specialization in Higher Secondary Education (like arts, science, commerce) 
# - degree_p: Degree Percentage 
# - degree_t: Under Graduation(Degree type)- Field of degree education 
# - workex: Work Experience (yes or no) 
# - etest_p: Employability test percentage (conducted by college) 
# - specialisation: Post Graduation(MBA)- Specialization 
# - mba_p: MBA percentage 
# - status: Status of placement- Placed/Not placed 
# - salary: Salary offered by corporate to candidates

# %% 
# Reading in the dataset:
#
# Source: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement?resource=download
# Here's also the raw link: https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv 
placement = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Analyzing the info of the dataset:
placement.info()

# %% [markdown]
# #### Potential Issues/Questions Dataset can Address

# %% [markdown]
# - Predicting the placement of student (whether they're placed or not). What sort of
# factors are assoicated whether a student gets placed? Do undergrad or MBA score matter
# more? How much does work experience play a role in placement? Could we make early predictions
# (like in high school) about who will be placed? Predicting placement will be a classification 
# problem.
#
# - Predicting the salary of student. In the dataset, only students that are placed receieve
# a salary, which leads to the column having plenty of missing values. However, we could still
# examine different features that may impact these salaries. For example, does the employability 
# test matter more than academic results? Predicting salaries will be a regression problem.
#
# - How much of a role does academics play a role in a student's employability? 
#
# - How important is one's degree type in determining placement and salary?




# %% [markdown]
# ### Step 2: Data Preparation

# %% [markdown]
# #### Generic Question Dataset can Address:
#
# How can we predict whether a student will be placed in a job or not?
# 
#
# #### Target Variable:
# Placement status (classification problem)
#
# This is already a column in the data, so there's no need to create one.
#
# #### Independent Business Metric:
# One independent business metric for this problem could simply be the placement
# rate of students from the college. Colleges can benefit financially with their 
# students finding jobs after their graduation because it's an indiactor of how the 
# college in doing with regards to providing pathways for jobs. Therefore, placement rate
# of a college (or their salaries) can be used to evaluate this classification model.
# 
# A metric that might even be more impactful is the placement rate of "high-risk" students
# determined by the model. If the model can predict early the liklihood of job placement, 
# then relevant resources can be allocated to help students who are not on track to get a job 
# after college. This metric would show how the model chooses students who need support and allows
# people to take appropriate action to help. 
#

# %%
# #### Dropping Unneeded Variables

# %%
# The only column that's evidently reasonable to drop is the serial number column.

placement = placement.drop(columns='sl_no')

# Verifying:
placement.info()
# We see that the serial number column was successfully dropped


# %% [markdown]
# #### Prevelance of Target Variable


# %%
# This is a classification problem, so we need to find the percentage of the data that
# is in the positive class, meaning placed students. 

# Number of Placed Students
placed_count = (placement['status'] == 'Placed').sum()

# Total number of students
total = len(placement)

# Prevelance
prevelance = placed_count/total
print(prevelance)

# The prevelance is 0.688%, which checks out since Data Wrangler shows the same value (69%)


# %% [markdown]
# #### Correcting Variable Types

# %%
# Changing categorical columns to the "category" data type:
cols = ["gender", "ssc_b", "hsc_b","hsc_s","degree_t","workex",
        "specialisation","status"]
placement[cols] = placement[cols].astype('category')

# Checking to see if the data type changed:
placement.dtypes

# %% [markdown]
# #### Collapsing Factor Levels

# %%
# Looking through all the category columns, it doesn't look like we need to do
# any collapsing because each column has so few categories:
#
# - gender: 2 (M or F)
# - ssc_b: 2 ("Others" or "Central")
# - hsc_b: 2 ("Others" or "central")
# - hsc_s: 3 ("Commerce", "Science", or "Arts")
# - degree_t: 3 ("Comm&Mgmt", "Sci&Tech", or "Others")
# - workex: 2 ("No" or "Yes")
# - specialisation: 2 ("Mkt&Fin" or "Mkt&HR")
# - status: 2 ("Placed" or "Not Placed")


# %% [markdown]
# #### One-Hot Encoding

# %%
# Let's one change each category column to have its own binary indicator column.
#
category_list = list(placement.select_dtypes('category'))   # list of all category vars

# We don't want to encode our target variable since out model would need it
# in one column.
category_list.remove('status')

# creating new binary columns for each category 
placement_encoded = pd.get_dummies(placement, columns=category_list)
placement_encoded.columns  # Verify



# %% [markdown]
# #### Normalizing Continuous Variables

# %%
# First we select all the numeric columns
numeric_cols = list(placement_encoded.select_dtypes('number'))

# Applying MinMax Scaler:
placement_encoded[numeric_cols] = MinMaxScaler().fit_transform(placement_encoded[numeric_cols])

# Verifying --> the min for each numeric column is 0 and the max is 1
placement_encoded[numeric_cols].describe()


# %% [markdown]
# #### Data Partitions


# %%
# Since salary isn't a predictive feature, we can drop it (since
# only placed students get a salary)

placement_clean = placement_encoded.drop(columns='salary')


# %%
# First split - Separating training data from the rest
# 55% of data will go to training

train, test = train_test_split(
    placement_clean,
    train_size=0.55,
    stratify=placement_clean['status'],
    random_state=42    # random_states fixed the random number generator so code is reproducible
)

# Looking at shapes:
print(f"Training set shape: {train.shape}")
print(f"Remaining set shape: {test.shape}")


# %%
# Second split - Split remaining data into tuning and test sets (50/50)

tune, test = train_test_split(
    test,
    train_size=0.5,
    stratify=test['status'],
    random_state=42
)


# Looking at shapes:
print(f"Tuning set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")

# %%
# Now we can check the prevelance of placed students in the overall dataset
# compared to the training, tuning, and testing datasets. We can do this using
# value_counts(normalize=True).

print("Overall prevalence:")
print(placement_clean['status'].value_counts(normalize=True))

print("\nTraining set prevalence:")
print(train['status'].value_counts(normalize=True))

print("\nTuning set prevalence:")
print(tune['status'].value_counts(normalize=True))

print("\nTest set prevalence:")
print(test['status'].value_counts(normalize=True))

# The distribution of placed vs not placed is essentially the same for all!



# %% [markdown]
# ### Step 3: Reflection

# %%
