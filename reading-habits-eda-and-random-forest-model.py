#!/usr/bin/env python
# coding: utf-8

# # Reading Habits EDA and Random Forest Model

# This is my beginner attempt at performing EDA and fitting a Random Forest Classification model on the Reading Habits dataset.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_palette("pastel")


# In[2]:


df = pd.read_csv('/kaggle/input/reading-habits-and-mood-impact-dataset/sleep and psychological effects.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


print(f'The table has {df.shape[0]} rows and {df.shape[1]} columns.')


# In[6]:


df.describe()


# In[7]:


#visualize the genre of favorite books distribution
book_genres = list(df['Favorite_Book_Genre'].unique())
genre_count = list(df['Favorite_Book_Genre'].value_counts().values)

plt.pie(data=df, labels=book_genres, x=genre_count, autopct='%1.1f%%')
plt.title('Distribution of Book Genres')
plt.show()


# In[8]:


print(f'Age range of readers is between {df["Age"].min()} and {df["Age"].max()}')


# In[9]:


#Defining 3 age groups for easier data analysis - low (18 - 25), medium (26 - 35), high (36 - 50)
low_age_group = df[(df['Age'] >= 18) & (df['Age'] <= 25)]
medium_age_group = df[(df['Age'] >= 26) & (df['Age'] <= 35)]
high_age_group = df[(df['Age'] >= 36) & (df['Age'] <= 50)]


# In[10]:


df['Favorite_Book_Genre'].unique() #Each of the genres in the dataframe


# In[11]:


#Creating a clustered bar plot to analyze the favorite genre by each age group

#Get favorite genre counts for each age group
low_age_genre_counts = low_age_group['Favorite_Book_Genre'].value_counts()
medium_age_genre_counts = medium_age_group['Favorite_Book_Genre'].value_counts()
high_age_genre_counts = high_age_group['Favorite_Book_Genre'].value_counts()

#Get the unique genres from the dataset
genres = df['Favorite_Book_Genre'].unique()

#DataFrame to combine all the counts
genre_counts_df = pd.DataFrame({
    '18-25': low_age_genre_counts,
    '26-35': medium_age_genre_counts,
    '36-50': high_age_genre_counts
}).fillna(0) #Fill in 0 in case no members of an age group select one of the genres

genre_counts_df = genre_counts_df.reset_index()
genre_counts_df.columns = ['Genre', '18-25', '26-35', '36-50']

plt.figure(figsize=(12, 6))
melted_df = genre_counts_df.melt(id_vars='Genre', var_name='Age Group', value_name='Count')

#Create the clustered bar plot
sns.barplot(x='Genre', y='Count', hue='Age Group', data=melted_df, edgecolor='black', palette='Set1')

plt.title('Favorite Book Genre by Age Group')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)  #Rotate genre names for readability
plt.legend(title='Age Group')

plt.tight_layout()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

#Ages are between 18 - 50
#Group Ages into 3 categories and identify each groups average reading time (hours) - low, medium, high
low_age_reading_average = low_age_group['Weekly_Reading_Time(hours)'].mean()
medium_age_reading_average = medium_age_group['Weekly_Reading_Time(hours)'].mean()
high_age_reading_average = high_age_group['Weekly_Reading_Time(hours)'].mean()

# Calculate the average reading time by age
average_reading_time = df.groupby('Age')['Weekly_Reading_Time(hours)'].mean()

reading_averages = [low_age_reading_average, medium_age_reading_average, high_age_reading_average]

#Plotting the results
plt.figure(figsize=(6, 6))
sns.barplot(x=['18 - 25', '26-35', '36-50'], y=reading_averages,
            edgecolor='black', color='dodgerblue', width=0.2)

plt.title('Average Time Spent Reading by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Reading Time (hours)')
plt.show()


# I would not have expected average reading time to go down with age...

# In[13]:


categorical_rows = df.select_dtypes(include=['object']).columns
print(f'Categorical columns are: \n{list(categorical_rows)}\n')

#Determine cardinality for all cat columns
for i in list(categorical_rows):
    print(f'The Cardinality of the {i} column is: {len(df[i].unique())}')


# All categorical columns have low cardinality - Going to try one hot encoding for all

# In[14]:


from sklearn.preprocessing import OneHotEncoder

#One hot encoding the gender and favorite book genre columns
oh_encoder = OneHotEncoder()
encoded_data = oh_encoder.fit_transform(df[['Gender', 'Favorite_Book_Genre']])
encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data, columns=oh_encoder.get_feature_names_out(['Gender', 'Favorite_Book_Genre']))

completed_df = pd.merge(df, encoded_df, left_index=True, right_index=True)
completed_df = completed_df.drop(columns=['Gender', 'Favorite_Book_Genre'])

print(completed_df.columns) # remaining columns are encoding and dropping the originals


# In[15]:


completed_df['Mood_Impact'].unique() #Verifying the unique values of Mood Impact


# In[16]:


#Label encode the Mood Impact Column before assigning it to our y variable
#Using label encoding as the Mood Impact is ordered (I believe)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
completed_df['Mood_Impact_encoded'] = le.fit_transform(completed_df['Mood_Impact'])

print(completed_df[['Mood_Impact', 'Mood_Impact_encoded']].head(10)) # Check encoding to ensure it worked


# In[17]:


completed_df.columns


# In[18]:


completed_df.drop(columns=['Mood_Impact'], inplace=True) #Drop cat column after encoding


# In[19]:


#Analyze correlation with a heatmap
y = completed_df['Mood_Impact_encoded']
X = completed_df.drop(columns=['Mood_Impact_encoded'])

plt.figure(figsize=(10,10))
sns.heatmap(data=X.corr(), annot=True)


# In[20]:


print(X.dtypes) #Got an error regarding Sparse Matrices, checking datatypes


# In[21]:


#Converting all the Sparse Matrix datatypes back to float64 dtypes - this code is ugly but yea...
X['Gender_f'] = X['Gender_f'].sparse.to_dense()
X['Gender_m'] = X['Gender_m'].sparse.to_dense()
X['Favorite_Book_Genre_Biography'] = X['Favorite_Book_Genre_Biography'].sparse.to_dense()
X['Favorite_Book_Genre_Fantasy'] = X['Favorite_Book_Genre_Fantasy'].sparse.to_dense()
X['Favorite_Book_Genre_Fiction'] = X['Favorite_Book_Genre_Fiction'].sparse.to_dense()
X['Favorite_Book_Genre_History'] = X['Favorite_Book_Genre_History'].sparse.to_dense()
X['Favorite_Book_Genre_Romance'] = X['Favorite_Book_Genre_Romance'].sparse.to_dense()
X['Favorite_Book_Genre_Science'] = X['Favorite_Book_Genre_Science'].sparse.to_dense()
X['Favorite_Book_Genre_Self-Help'] = X['Favorite_Book_Genre_Self-Help'].sparse.to_dense()

print(X.dtypes)


# In[22]:


#Splitting training and testing data, using 0.1 split due to very limited dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[23]:


print(X_train.dtypes)


# In[24]:


#Random Forest Test - not using any parameters to get a baseline accuracy score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


# In[25]:


#Tuning hyperparameters using GridSearchCV method
from sklearn.model_selection import GridSearchCV

#Added a small number of options for each parameters (over utilized kaggle notebook, so reduced number)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_leaf_nodes': [None],
    'max_features': ['sqrt']
}

rf = RandomForestClassifier(random_state=42)

#Using verbose zero to reduce clutter
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)


# This model might be overfitted.. But still good practice I think.
