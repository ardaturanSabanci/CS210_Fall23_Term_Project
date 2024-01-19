#!/usr/bin/env python
# coding: utf-8

# # ARDA TURAN CS210 FALL'23 PROJECT
# ***
# My project is aiming to give some valuable information about my listening habits and taste of songs while searching for a question that I have in my mind. This question is about the genres of musics that I put in my playlists. Also, I have got a playlist called 'Various' and it contains 389 songs from different artist, languages, etc. Additionally, I used this playlist as my project material because the importance of data has been significantly getting important and becoming widespread, especially after the pandemics. That is why I collected my data from a Spotify analyzer by scraping it. Also, I collected more general data about my account by requesting it from Spotify. Shortly, I used them in order to test my hypothesis, and see the shape of my data. I used different tecniques such as exploratory data analysis, various data visualization, and machine learning algorithms. In addition to this, I tested my null hypothesis by determining the p-value and comparing it with the significance level, so that I can whether reject it or not. I found out that my data is rational since I could be able to find the correct answer about rejecting or failing to reject my hypothesis correlating with my data. Additionally, I was shocked about the deviation, diversity, and number of types of genres. Other than that, I am quite happy with the results because of the fact that I see have a taste of music related to my moods. The distibution of my data is suitable for data projects since it does not depend on stereotypes.
# ***
# Link of my presentation document:
# ***

# ## Importing the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import bs4
import requests
from scipy.stats import binom_test
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#import spotipy
#from spotipy.oauth2 import SpotifyClientCredentials
#client_credentials_manager = SpotifyClientCredentials(client_id='491862eccb624461a90a6ec9a26c94d7', client_secret='342e7a81e01241799d0ff4a6319eaf82')
#sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# ## Scraping my data

# In[2]:


file_path = r'/Users/ardaturan/Desktop/index.html'

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

soup = bs4.BeautifulSoup(html_content, 'html.parser')
pretty_html = soup.prettify()


# ***

# In[3]:


music_blocks = soup.findAll('tr', {'data-id-song': True})


# In[4]:


def scrap_page(music_blocks):
    musicb_data = {}
    musicb_data["#"] = []
    musicb_data["Song"] = []
    musicb_data["Artist"] = []
    musicb_data["Popularity"] = []
    musicb_data["BPM"] = []
    musicb_data["Genres"] = []
    musicb_data["Parent Genres"] = []
    musicb_data["Album"] = []
    musicb_data["Album Date"] = []
    musicb_data["Time"] = []
    musicb_data["Dance"] = []
    musicb_data["Energy"] = []
    musicb_data["Acoustic"] = []
    musicb_data["Instrumental"] = []
    musicb_data["Happy"] = []
    musicb_data["Speech"] = []
    musicb_data["Live"] = []
    musicb_data["Loud"] = []
    musicb_data["Key"] = []
    musicb_data["Time Signature"] = []
    musicb_data["Added at"] = []
    musicb_data["Spotify Track Id"] = []
    musicb_data["Album Label"] = []
    musicb_data["Camelot"] = []
    musicb_data["Spotify Track Img"] = []
    musicb_data["Song Preview"] = []
    
    
    for music in music_blocks:
        music_index = music.find("td", {"class" : "td-number text-center"}).get_text()
        try:
            musicb_data["#"].append(music_index)
        except:
            musicb_data["#"].append("None")
        
        music_name = music.find("td", {"class" : "td-name"}).get_text()
        try:
            musicb_data["Song"].append(music_name)
        except:
            musicb_data["Song"].append("None")
            
        music_img = music.findAll("img", {"class" : "track-table-img"})
        for img in music_img:
            if img.get("src") is not None:
                image = img["src"]
        try:
            musicb_data["Spotify Track Img"].append(image)
        except:
            musicb_data["Spotify Track Img"].append("None")
            
        music_pre = music.findAll('div', {'class': 'track-list-item-right suggest-player-table suggest-player'})
        for pre in music_pre:
            if pre.get('data-previewurl') is not None:
                link = pre['data-previewurl']
        try:
            musicb_data["Song Preview"].append(link)
        except:
            musicb_data["Song Preview"].append("None")
        
        music_artist = music.find("td", {"class" : "td-number oferflow"}).get_text()
        try:
            musicb_data["Artist"].append(music_artist)
        except:
            musicb_data["Artist"].append("None")

        music_popularity0 = music.findAll("td", {"class" : "td-number text-center"})
        music_popularity = music_popularity0[1].get_text()
        try:
            musicb_data["Popularity"].append(music_popularity)
        except:
            musicb_data["Popularity"].append("None")

        music_BPM0 = music.findAll("td", {"class" : "td-number text-center"})
        music_BPM = music_BPM0[2].get_text()
        try:
            musicb_data["BPM"].append(music_BPM)
        except:
            musicb_data["BPM"].append("None")

        music_genres0 = music.findAll("td", {"class" : "td-number oferflow"})
        music_genres = music_genres0[1].get_text()
        try:
            musicb_data["Genres"].append(music_genres)
        except:
            musicb_data["Genres"].append("None")

        music_parent_genre0 = music.findAll("td", {"class": "td-number oferflow"})
        music_parent_genre = music_parent_genre0[2].get_text()
        try:
            musicb_data["Parent Genres"].append(music_parent_genre)
        except:
            musicb_data["Parent Genres"].append("None")

        music_album0 = music.findAll("td", {"class": "td-number oferflow"})
        music_album = music_album0[3].get_text()
        try:
            musicb_data["Album"].append(music_album)
        except:
            musicb_data["Album"].append("None")

        music_album_date0 = music.findAll("td", {"class" : "td-number text-center"})
        music_album_date = music_album_date0[3].get_text()
        try:
            musicb_data["Album Date"].append(music_album_date)
        except:
            musicb_data["Album Date"].append("None")

        music_time0 = music.findAll("td", {"class" : "td-number text-center"})
        music_time = music_time0[4].get_text()    
        try:
            musicb_data["Time"].append(music_time)
        except:
            musicb_data["Time"].append("None")

        music_dance0 = music.findAll("td", {"class" : "td-number text-center"})
        music_dance = music_dance0[5].get_text()  
        try:
            musicb_data["Dance"].append(music_dance)
        except:
            musicb_data["Dance"].append("None")

        music_energy0 = music.findAll("td", {"class" : "td-number text-center"})
        music_energy = music_energy0[6].get_text()    
        try:
            musicb_data["Energy"].append(music_energy)
        except:
            musicb_data["Energy"].append("None")

        music_acoustic0 = music.findAll("td", {"class" : "td-number text-center"})
        music_acoustic = music_acoustic0[7].get_text()    
        try:
            musicb_data["Acoustic"].append(music_acoustic)
        except:
            musicb_data["Acoustic"].append("None")

        music_instrumental0 = music.findAll("td", {"class" : "td-number text-center"})
        music_instrumental = music_instrumental0[8].get_text()    
        try:
            musicb_data["Instrumental"].append(music_instrumental)
        except:
            musicb_data["Instrumental"].append("None")

        music_happy0 = music.findAll("td", {"class" : "td-number text-center"})
        music_happy = music_happy0[9].get_text()    
        try:
            musicb_data["Happy"].append(music_happy)
        except:
            musicb_data["Happy"].append("None")

        music_speech0 = music.findAll("td", {"class" : "td-number text-center"})
        music_speech = music_speech0[10].get_text()    
        try:
            musicb_data["Speech"].append(music_speech)
        except:
            musicb_data["Speech"].append("None")

        music_live0 = music.findAll("td", {"class" : "td-number text-center"})
        music_live = music_live0[11].get_text()    
        try:
            musicb_data["Live"].append(music_live)
        except:
            musicb_data["Live"].append("None")

        music_loud0 = music.findAll("td", {"class" : "td-number text-center"})
        music_loud = music_loud0[12].get_text()     
        try:
            musicb_data["Loud"].append(music_loud)
        except:
            musicb_data["Loud"].append("None")

        music_key0 = music.findAll("td", {"class" : "td-number text-center"})
        music_key = music_key0[13].get_text()     
        try:
            musicb_data["Key"].append(music_key)
        except:
            musicb_data["Key"].append("None")

        music_time_signature0 = music.findAll("td", {"class" : "td-number text-center"})
        music_time_signature = music_time_signature0[14].get_text()     
        try:
            musicb_data["Time Signature"].append(music_time_signature)
        except:
            musicb_data["Time Signature"].append("None")

        music_added0 = music.findAll("td", {"class" : "td-number text-center"})
        music_added = music_added0[15].get_text()    
        try:
            musicb_data["Added at"].append(music_added)
        except:
            musicb_data["Added at"].append("None")

        music_spotify_trackid0 = music.findAll("td", {"class" : "td-number text-center"})
        music_spotify_trackid = music_spotify_trackid0[16].get_text()      
        try:
            musicb_data["Spotify Track Id"].append(music_spotify_trackid)
        except:
            musicb_data["Spotify Track Id"].append("None")

        music_album_label0 = music.findAll("td", {"class" : "td-number text-center"})
        music_album_label = music_album_label0[17].get_text()      
        try:
            musicb_data["Album Label"].append(music_album_label)
        except:
            musicb_data["Album Label"].append("None")

        music_camelot0 = music.findAll("td", {"class" : "td-number text-center"})
        music_camelot = music_camelot0[18].get_text()      
        try:
            musicb_data["Camelot"].append(music_camelot)
        except:
            musicb_data["Camelot"].append("None")

    return musicb_data


# In[5]:


def scrape_m_page(music_blocks):
    page_music_data = []
    
    num_blocks = len(music_blocks)
    
    for idx in range(num_blocks):
        page_music_data.append(scrape_mblock(music_blocks[idx]))
        
    return page_music_data


# In[6]:


#base_scraping_link = r'/Users/ardaturan/Desktop/SABANCI/Sophomore 2/CS 210/HW1/CS210 - Homework 1 - for BeautifulSoup.html'

music_list = []

music = scrap_page(music_blocks)
df = pd.DataFrame(music)

df = df.rename(index = lambda x: x + 1)

df


# ***
# # Exploring my Dataset (EDA) & Preprocessing

# ***
# Let's learn more about my dataset. In this part, you will see the shape of my dataset.

# In[7]:


print("Number of columns (features) are:", df.shape[1])
print("Number of rows (samplings) are:", df.shape[0])


# In[61]:


df.shape


# ***

# In[8]:


df.describe().transpose()


# ***

# In[9]:


print("Each columns' non-null counts and their data types: \n")

df[['#', 'Popularity', 'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Time Signature']] = df[['#', 'Popularity', 'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Time Signature']].apply(pd.to_numeric)
df['Loud'] = df['Loud'].str.extract('(-?\d+)').astype(int)

df.info()


# ***
# Checking if there is any null row.
# ***
# **Fact:** You see there is no empty row which is a good feature in terms of avoiding the one from getting weird results when implementing Machine Learning models.

# In[10]:


print("Getting how many rows are null for each feature: \n", df.isna().sum())


# Here is the variable names.

# In[11]:


print("Variable names: ")
print(df.columns.tolist(), "\n")


# ***
# I removed 'Spotify Track Img' and 'Song Preview' columns since they have no contribution to ordering.

# In[12]:


df.drop(['Spotify Track Img'], axis = 1, inplace=True)
df.drop(['Song Preview'], axis = 1, inplace=True)

df.head()


# ***
# Since I have finished investigating recurring or null values, I move on with the most interesting part, which is the sorting.
# ***
# First, I start with the most popular songs.

# In[13]:


a= df.sort_values('Popularity',ascending=False)[:10]
a[['Song','Popularity','Artist','Genres']]


# ***
# Now, the least popular ones.

# In[14]:


a= df.sort_values('Popularity',ascending=True)[0:10]
a[['Song','Popularity','Artist', 'Genres']]


# ***
# This is about our hypothesis.

# In[15]:


df['Genres'] = df['Genres'].str.split(', ')
genre_counts = df.explode('Genres')['Genres'].value_counts()

top_10_genres = genre_counts.head(10)
top_10_genres


# ***
# These are the top genres of my playlist but lets see whether they are also in the list that is ordered with respect to the popularity.

# In[16]:


df_exploded = df.explode('Genres')

average_popularity = df_exploded.groupby('Genres')['Popularity'].agg(['mean', 'size'])

popular_genres = average_popularity[average_popularity['size'] > 0]

popular_genres = popular_genres.rename(columns={'mean': 'Popularity Score'})

top_genres = popular_genres.nlargest(10, 'Popularity Score')

print("Top 10 Genres with Average Popularity Score:")
top_genres[['Popularity Score']]


# ***
# Now here you see my best song. I hope you like it.

# In[17]:


for i in range(len(music["Song"])):
    music["Song"][i] = music["Song"][i].lstrip("  ")

df = pd.DataFrame(music)
df = df.rename(index = lambda x: x + 1)
df[['Popularity', 'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Time Signature']] = df[['Popularity', 'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Time Signature']].apply(pd.to_numeric)
df['Loud'] = df['Loud'].str.extract('(-?\d+)').astype(int)

#df['Popularity_Category'] = df['Popularity'].apply(categorize_popularity)
#df['BPM_Category'] = df['BPM'].apply(categorize_bpm)

filtered_df = df[df.Song == "Die For You"]
filtered_df


# ***
# With this code, you can search for the name of a song in a specific line.

# In[18]:


df[['Artist']].iloc[1]


# ***
# Last but not least, here is the most common album.

# In[19]:


album_counts = df['Album'].value_counts()

most_common_album = album_counts.idxmax()

print(f"The name of the most common album among tracks: ")

most_common_album


# In[20]:


def categorize_happy(happy):
    if happy < 30:
        return 'Miserable'
    elif 30 <= happy <= 70:
        return 'Average'
    else:
        return 'Joyful'
    
def categorize_popularity(popularity):
    if popularity > df['Popularity'].quantile(0.75):
        return 'High'
    elif popularity >= df['Popularity'].quantile(0.25):
        return 'Medium'
    else:
        return 'Low'
    
#def categorize_genres(genres):
 #   if genres in ['hip hop', 'pop', 'electronic', 'jazz', 'r&b', 'latin', 'classical', 'easy listening', 'world/traditional', 'folk/accoustic', 'new age', 'country', 'metal' ]:
  #      return True
   # else:
    #    return False
    
df['Popularity_Category'] = df['Popularity'].apply(categorize_popularity)

df['Popularity_Category'] = df['Popularity'].apply(categorize_popularity)
df['Happiness_Category'] = df['Happy'].apply(categorize_happy)
#df['Genres_Category'] = df['Genres'].apply(categorize_genres)

df['Popularity_Category'].value_counts(), df['Happiness_Category'].value_counts()#, df['Genres_Category'].value_counts()


# # Data Visualization and Analysis

# I use scattered plot to see the deviation of my data.

# In[21]:


plt.figure(figsize=(8, 6))

plt.scatter(df['Loud'], df['Energy'], alpha = 0.7)

plt.title('Relationship between Energy and Loudness')
plt.xlabel('Loudness')
plt.ylabel('Energy')

#plt.xlim(-20, -2)
#plt.ylim(0, 100)

plt.grid(True)

plt.show()    


# Now, I plot my special categories named as Happiness, Popularity category.

# In[22]:


fig, axs = plt.subplots(1, 2, figsize=(12, 8))

df['Happiness_Category'].value_counts().plot(kind="bar", color="blue", ax = axs[1])
axs[1].set_title('Happiness')
axs[1].set_ylabel('Frequencies')
axs[1].set_xlabel('Happy')
axs[1].set_ylim([0, 300])

df['Popularity_Category'].value_counts().plot(kind="bar", color="green", ax = axs[0])
axs[0].set_title('Popularities')
axs[0].set_ylabel('Frequencies')
axs[0].set_xlabel('Popularity')
axs[0].set_ylim([0, 300])

plt.tight_layout()

plt.show()


# I assign a new variable again to have a better visualization examples over my data.

# In[23]:


df['Album Date'] = pd.to_datetime(df['Album Date'], errors='coerce')

df['Month'] = df['Album Date'].dt.month

avg_popularity_by_month = df.groupby('Month')['Popularity'].mean()

df.head()


# Now, I use our new variable to see the monthly distribution.

# In[24]:


custom_month_order = [calendar.month_abbr[i] for i in range(1, 13)]

plt.figure(figsize=(10, 6))
plt.plot(avg_popularity_by_month.index, avg_popularity_by_month.values, marker='o', linestyle='-', color = "red")

plt.xlabel('Month')
plt.ylabel('Average Popularity')
plt.title('Average Popularity of Music Albums Across Months')
plt.xticks(range(1, 13), custom_month_order)  # Use actual month numbers for x-axis ticks

plt.ylim(avg_popularity_by_month.min() - 5, avg_popularity_by_month.max() + 5)

plt.xticks(rotation=45)
plt.grid(True)

plt.show()


# Here is the barplot of top genres according to their popularity levels.

# In[25]:


sns.set_style(style='darkgrid')
plt.figure(figsize=(8,4))
Top = df.sort_values('Popularity', ascending=False)[:10]
sns.barplot(y = 'Genres', x = 'Popularity', data = Top).set(title='Top Genres by Popularity')


# Also the genres with the least popularities.

# In[26]:


sns.set_style(style='darkgrid')
plt.figure(figsize=(8,4))
Top = df.sort_values('Popularity', ascending=True)[:10]
sns.barplot(y = 'Genres', x = 'Popularity', data = Top).set(title='Top Genres by Popularity')


# In[27]:


df[['#', 'Popularity', 'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Time Signature']] = df[['#', 'Popularity', 'BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Time Signature']].apply(pd.to_numeric)

numerical_columns = df.select_dtypes(include = ['int64', 'float64'])
correlation_matrix = numerical_columns.corr()

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='inferno', linewidths=.5, ax=ax)

plt.title("Correlation Matrix")
plt.show()


# In[28]:


expanded_genres = df['Genres'].str.split(', ').explode()
top_10_genres = expanded_genres.value_counts().head(10).index

# Creating a new dataframe that maps songs to each genre they belong to
genre_mappings = expanded_genres.reset_index().merge(df[['Energy', 'Dance']], left_on='index', right_index=True)
genre_mappings = genre_mappings[genre_mappings['Genres'].isin(top_10_genres)]

aggregated_values = genre_mappings.groupby('Genres').agg({'Energy': 'sum', 'Dance': 'sum'})
plt.figure(figsize=(10, 6))
aggregated_values.plot(kind='bar', stacked=True, color=['#669bbc', '#003049'], figsize=(10, 6))
plt.title('Energy and Dance Values by Top 10 Genres')
plt.xlabel('Genres')
plt.ylabel('Value')
#plt.xticks(rotation=45)
plt.show()


# In[29]:


filtered_artists = {}
filtered_artists["Artist"] = []
filtered_artists["Happy"] = []

for i in range(len(df)):
    if df['Artist'].iloc[i] == 'Adele' or df['Artist'].iloc[i] == 'Travis Scott' or df['Artist'].iloc[i] == 'The Weeknd':
        filtered_artists['Artist'].append(df['Artist'].iloc[i])
        filtered_artists['Happy'].append(df['Happy'].iloc[i])
        
filtered_artists_df = pd.DataFrame(filtered_artists)
filtered_artists_df = filtered_artists_df.rename(index = lambda x: x + 1)
#filtered_artists_df['Popularity_Category'] = filtered_artists_df['Popularity'].apply(categorize_popularity)
filtered_artists_df['Happiness_Category'] = filtered_artists_df['Happy'].apply(categorize_happy)

plt.figure(figsize = (8, 6))
sns.countplot(x = "Happiness_Category", hue = "Artist", data = filtered_artists_df, palette = "inferno", order = ["Miserable", "Average", "Joyful"])
#plt.hist(filtered_artists_df[['Artist', 'BPM_Category']], density=True, histtype='bar', color = ["red", "blue"], label=artists)

plt.legend(prop = {'size': 12})
plt.title('Happiness Levels in Music of Adele, Travis Scott, and The Weeknd')
plt.xlabel("Happiness")
plt.ylabel('Frequencies')
plt.show()


# # Hypothesis Testing

# My null Hypothesis: I do not listen to music from at least 10 different genres.
# ***
# Alternative Hypothesis: I listen to music from at least 10 different genres.

# In[ ]:


genres = df['Genres']

num_unique_genres = genres.nunique()

threshold_genres = 10

p_value = binom_test(num_unique_genres, n=len(genres), p=0.5)

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
sns.countplot(y='Genres', data=df, order=df['Genres'].value_counts().index)
plt.title('Genre Distribution')
plt.xlabel('Count')
plt.ylabel('Genres')

plt.text(0.5, -1, f'P-value: {p_value:.4f}', ha='center', va='center', fontsize=12, color='red')

#plt.savefig('genre_distribution.png')
plt.show()


# In[31]:


genres = df['Genres']

num_unique_genres = genres.nunique()

threshold_genres = 100

p_value = binom_test(num_unique_genres, n=len(genres), p=0.5)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. You listen to music from at least 10 different genres.")
else:
    print("Fail to reject the null hypothesis. You may not listen to music from at least 10 different genres.")

print("P-value:", p_value)


# # Machine Learning

# I mapped columns that are categorical, into numerical data with LabelEncoder library due to the fact that ML systems may fail.
# 

# In[32]:


le = LabelEncoder()

for col in df.columns:
  if col == 'Song' or col == 'Artist' or col == 'Genres' or col=='Parent Genres' or col == 'Album' or col == 'Album Date' or col == 'Time' or col == 'Key' or col == 'Time Signature' or col == 'Added at' or col == 'Spotify Track Id' or col == 'Album Label' or col == 'Camelot' or col == 'Spotify Track Img' or col == 'Song Preview' or col == 'Popularity_Category' or col == 'Happiness_Category' or col == 'Month':
    le.fit(df[col])
    df[col] = le.transform(df[col])
    
df.head()
     


# ***
# I dropped some columns that I cannot use.

# In[33]:


df_ml = df.drop('Spotify Track Img', axis=1)


# In[34]:


df_ml = df.drop('Song Preview', axis=1)


# In[35]:


df_ml = df.drop('Album Date', axis=1)


# In[36]:


df_ml = df.drop('Key', axis=1)


# In[37]:


df_ml = df.drop('Time', axis=1)


# In[38]:


df_ml = df.drop('Added at', axis=1)


# In[39]:


df_ml.head()


# #Supervised learning

# In[40]:


missing_values = df.isnull().sum()
print("Missing values: ")
print(missing_values, "\n")

for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

# Split the data into training and test sets (80% training, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[41]:


shuffled_df = shuffle(df, random_state=42)  # Shuffle the DataFrame

# Independent variables (X) and dependent variable (y)
#X = shuffled_df.drop(columns=['health_metrics'], axis=1)

X = shuffled_df.drop('Genres', axis=1)        # X contains all columns except 'health_metrics', .drop('health_metrics', axis=1)
y = shuffled_df['Genres']

# Splitting into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[44]:


plt.figure(figsize=(8,4))
sns.regplot(data= df, y='Popularity', x='BPM', color='#054907').set(title='Regression Plot - Popularity vs BPM Correlation')


# # Extra 
# As I stated before, the data that I received from Spotify has no information about genre of a song. In this part, I used that Spotify data as an extra study on my project.

# ## EDA & Visualization

# In[118]:


import json

with open('Playlist1.json', 'r') as file:
    data = json.load(file)

playlists = data.get('playlists', [])
df = pd.DataFrame(playlists)


# In[119]:


df.head()


# In[120]:


df.shape


# In[121]:


print("Variable names: ")
print(df.columns.tolist(), "\n")


# In[122]:


df.describe()


# In[123]:


print("Getting how many rows are null for each feature: \n", df.isna().sum())


# In[124]:


df['lastModifiedDate'] = pd.to_datetime(df['lastModifiedDate'])


plt.bar(df['name'], df['lastModifiedDate'])
plt.xlabel('Playlist Name')
plt.ylabel('Last Modified Date')
plt.title('Last Modified Date of Playlists')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[125]:


df[['lastModifiedDate']] = df[['lastModifiedDate']].apply(pd.to_numeric)

numerical_columns = df.select_dtypes(include = ['int64', 'float64'])
correlation_matrix = numerical_columns.corr()

fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='inferno', linewidths=.5, ax=ax)

plt.title("Correlation Matrix")
plt.show()


# In[ ]:




