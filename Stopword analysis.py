# import libraries and stopword lists
import en_core_web_sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# gensim stopwords
gensim_sw = set(STOPWORDS)
gensim_sw_series = pd.Series(list(gensim_sw)).sort_values()

# nltk stopwords
nltk_sw = set(stopwords.words('english'))
nltk_sw_series = pd.Series(list(nltk_sw)).sort_values()

# sklearn
sklearn_sw = set(ENGLISH_STOP_WORDS)
sklearn_sw_series = pd.Series(list(sklearn_sw)).sort_values()

# SMART stopword list
with open('SMART_stopwords.txt', 'r') as f:
    lines = f.read().splitlines()
SMART_sw_series = pd.Series(lines).sort_values()

# spacy stopwords
nlp = en_core_web_sm.load()
spacy_sw = nlp.Defaults.stop_words
spacy_sw_series = pd.Series(list(spacy_sw)).sort_values()

# Stanford Core NLP stopword list
with open('coreNLP_stopwords.txt', 'r') as f:
    lines = f.read().splitlines()
coreNLP_sw_series = pd.Series(lines).sort_values()

# In R, there are many more lists available
# lexicon
# - Leveled Dolch list of 220 words
# - Fry's most commonly used words (25 to 1000)
# - Matthew-Jocker's expanded topic modelling list
# - Loughran-McDonald short and long
# - Lucerne
# - MALLET
# - Python

# stopwords
# - marimo
# - nltk
# - stopwords-iso

# tidytext
# - onix

# tm / quanteda tidytext / stopwords
# - smart
# - snowball

# concatenate all the lists together and determine the total number of unique stopwords
frames = [coreNLP_sw_series, gensim_sw_series, nltk_sw_series, sklearn_sw_series, SMART_sw_series, spacy_sw_series]
all_stopwords = pd.concat(frames)

all_unique_stopwords = set(all_stopwords)
print("Across the six lists, there are " + str(len(all_unique_stopwords)) + " unique stopwords")

# count the number of times each stopword appears
sw_freq = Counter(all_stopwords)
freq_df = pd.DataFrame.from_dict(sw_freq, orient='index', columns=['count']).sort_values(by=['count'], ascending=False)
freq_df

# define a function to add data labels to the bar charts
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i] + 5, y[i], ha='center')

# barchart of the total number of words in each stopword list
names = ['coreNLP', 'gensim', 'nltk', 'sklearn', 'SMART', 'spacy']
num_words = [coreNLP_sw_series.shape[0], len(gensim_sw), len(nltk_sw), len(sklearn_sw), SMART_sw_series.shape[0], len(spacy_sw)]

fig, axs = plt.subplots()
sns.barplot(x=names, y=num_words, palette='Blues_d')
addlabels(x=names, y=num_words)
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlabel("List source")
axs.set_ylabel("Number of items");

# calculate the number of words appearing one to six times
x = np.sort(freq_df['count'].unique())
y = freq_df.groupby(['count']).size().values

fig, axs = plt.subplots()
sns.countplot(data=freq_df, x='count', palette='Blues_d')
addlabels(x=x, y=y)
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlabel("Number of lists")
axs.set_ylabel("Frequency");
