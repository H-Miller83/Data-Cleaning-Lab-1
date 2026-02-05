# %% [markdown]
# # Step 4: Pipeline Functions

# %%
# Import Needed Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# %% [markdown] 
# ## College Completion Dataset Pipeline

# %%
# Loading In Data:
def college_data(path):
    return pd.read_csv(path)
# The path should be something like "data/cc_institution_details.csv"

# %%
# Dropping Unneeded Columns
def drop_unneeded_columns(df):
    columns_to_drop = [
        'index', 'unitid', 'site', 'nicknames',
        'long_x', 'lat_y', 'similar'
    ]
    return df.drop(columns=columns_to_drop)


# %%
# Creating Target Variable
def create_grad_rate(df):     
    df['grad_rate'] = df['grad_150_value'] / df['cohort_size']
    return df

# This function adds in the new target variable column.

# Then I'll drop the one row that has a grad_rate of a 100 even though it should
# be in between 0 and 1:

def clean_grad_rate(df):
    df = df.drop(index=208)
    return df


# %%
# Prevelance of Target Variable
#
# Prportion of colleges with a grad rate above 80%:
def grad_rate_prevalence(df, threshold=0.80):
    prevelance = (df['grad_rate'] > threshold).mean()
    return prevelance


# %%
# Correcting Variables Types
# --> Making Category dtypes:

def fix_dtypes(df):
    categorical_cols = ['state', 'level', 'control']
    df[categorical_cols] = df[categorical_cols].astype('category')
    return df

# %% 
# Collapsing Factor Levels
# --> Collapsing the state column

def collapse_states(df, top_states=None):
    if top_states is None:
        top_states = [
            "California",
            "New York",
            "Pennsylvania",
            "Texas",
            "Ohio",
        ]
    df["state"] = (
        df["state"]
        .apply(lambda x: x if x in top_states else "Other")
        .astype("category")
    )

    return df


# %%
# One Hot Encoding

def one_hot_encode(df):
    category_list = list(df.select_dtypes('category'))
    category_list.remove('state')  # keeping state unencoded (just level & control)
    return pd.get_dummies(df, columns=category_list)


# %%
# Normalizing Continuous Variables

def scale_numeric(df):
    numeric_cols = df.select_dtypes('number').columns
    df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    return df


# %%
# Data Partitions

def split_data(df, train_size=0.55, random_state=83):
    # getting the training data
    train, test = train_test_split(
        df, train_size=train_size, random_state=random_state
    )

    # getting the tuning/testing data
    tune, test = train_test_split(
        test, train_size=0.5, random_state=random_state
    )

    return train, tune, test




# %% [markdown]
# ## Job Placement Pipeline



# %%
# Loading In Data:
def placement_data(path):
    return pd.read_csv(path)
# The path should be something like "data/Placement_Data_Full_Class.csv"


# %%
# Dropping Unneeded Columns
def drop_unneeded_columns(df):
    df.drop(columns=["sl_no"]) 
    return df


# %%
# Prevelance of Target Variable
#
# Target Var = "Status" column

def placement_prevalence(df, target_col="status", positive_class="Placed"):
    return (df[target_col] == positive_class).mean()


# %%
# Correcting Variables Types
# --> Making Category dtypes:

def cast_categorical_columns(df):
    categorical_cols = [
        "gender", "ssc_b", "hsc_b", "hsc_s",
        "degree_t", "workex", "specialisation", "status"
    ]
    df[categorical_cols] = df[categorical_cols].astype("category")
    return df



# %% 
# One Hot Encoding
#
# Encoding all the categorical variables except the target variable
# (placement status)

def one_hot_encode_features(df, target_col="status"):
    categorical_cols = list(df.select_dtypes("category"))
    categorical_cols.remove(target_col)
    return pd.get_dummies(df, columns=categorical_cols)



# %%
# Normalizing Continuous Variables

def normalize_numeric_features(df):
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    return df



# %%
# Dropping the salary colunm (b/c it's associated w/ target variable)

def drop_salarys(df):
    return df.drop(columns=["salary"])


# %%
# Data Partitions


def split_train_tune_test(
        df, target_col="status",
        train_size=0.55, random_state=83):
  
    # Creating the training set:
    train, test = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    # Splitting into the tune and test set:
    tune, test = train_test_split(
        test,
        train_size=0.5,
        stratify=test[target_col],
        random_state=random_state,
    )

    return train, tune, test

