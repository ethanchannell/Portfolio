# Song Lyrics Over Time and Genre
Hello! This was a project that was completed during the Fall 2021 Semester.The project was completed in a team with divided responsiblities where I was responsible for creating the NLP model. The rest of this post will go through how and why our team decided to focus on this project along with the code utilized to train the model.

During this project, I learned about creating and improving an NLP model in Python and the steps for cleaning the data to be used in the data. This post gives a quick overview about the project, the model, and the visualization. For the full paper, you can access it [here](https://drive.google.com/file/d/1sZyXOaFHQJnksldIRIWnP1xHC62p2afa/view?usp=sharing) which will give more motivation, background, and steps taken in the project to get to the final output.

# Introduction
For our project, our team decided to focus on topic analysis on song lyrics over time and genre. We were curious to determine how topics of songs have changed over time and how different genres topic similiarity comparedIn ordered to do so, we gathered lyric, genre, and release time data from the Millions Song Dataset to gather an inital 550,000 songs. We then proceeded to clean the data using the code below. The data is cleaned so that it can be processed for an NLP model.

# Data Cleaning

```python
mport re
import numpy as np
import pandas as pd
from pprint import pprint
import time

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import HdpModel
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.phrases import Phrases

#Spacy for lemmatization
import spacy

#Plotting tools
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from collections import Counter
import logging
import os
# Combine NTLK stopwords with gensim's stopwords and add additional stopwords
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
stop_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustnt",'needn',"needn't",'shan',"shant",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't",'from','use','embedShare','urlcopycmbedcopy','right',"ain't",'to','get']
#stop_words.extend(['from', 'use','embedShare','urlcopycmbedcopy',"right","ain't","to","get"])
all_stopwords = gensim.parsing.preprocessing.STOPWORDS
all_stopwords_gensim = STOPWORDS.union(set(stop_words))

#Read in csv file with data
df = pd.read_csv('lyrics_with_year.csv')
#Converts df to string type for further analysis
df = df.astype("string")
df = df.fillna("")
print(len(df))
#Some data pulled is not actual lyric and long paragraphs of text so we filter this out through multiple steps
df = df[df["lyrics"].apply(lambda x: len(x) < 8000)]
print(len(df))
#Remove the ft. odd occurences cutoff is 4 or more occurences
df = df[df["lyrics"].apply(lambda x: x.count("ft.") < 3)]
print(len(df))
df = df.reset_index()
df = df.drop(columns = ["index"])
df = df.dropna()
print(len(df))

#Remove puncuation/lower casing and URL embeded words
df["lyrics_processed"] = df["lyrics"].map(lambda x: x.replace('4EmbedShare URLCopyEmbedCopy', ''))
df["lyrics_processed"] = df["lyrics_processed"].map(lambda x: x.replace('EmbedShare URLCopyEmbedCopy', ''))
df["lyrics_processed"] = df["lyrics_processed"].map(lambda x: x.replace('\n', ' '))
df["lyrics_processed"] = df["lyrics_processed"].map(lambda x: x.replace('REPEAT', ''))
df["lyrics_processed"] = df["lyrics_processed"].map(lambda x: re.sub("[\(\[].*?[\)\]]", '', x))
df["lyrics_processed"] = df["lyrics_processed"].map(lambda x: x.replace('CHORUS', ''))
df["lyrics_processed"] = df["lyrics_processed"].map(lambda x: re.sub('[,\.!?]','',x))

#Ensures that the df is not empty
df = df[df["lyrics_processed"] != '']

#Ensures that each string of lyrics has atleast 20 characets
df = df[df["lyrics_processed"].apply(lambda x: len(x) >= 20)]

#Makes lyrics all lower case
df["lyrics_processed"] = df['lyrics_processed'].str.lower()

#New df that will be without the lyrics column
df_2 = df
df_2 = df_2.drop(columns = ["lyrics"])
df_2.head()
print(len(df_2))
#Removes lyrics with too few characters and too many characters
df_2 = df_2[df_2["lyrics_processed"].apply(lambda x: len(x) > 400)]
df_2 = df_2[df_2["lyrics_processed"].apply(lambda x: len(x) < 5000)]
print(len(df_2))

#Detect the language of each song
df_2["language"] = df_2["lyrics_processed"].map(lambda x: detect(x))

#Filter to only include enlgish songs for now since stopwords depend on this
df_3 = df_2[df_2["language"] == "en"]
df_3 = df_3.reset_index()
df_3 = df_3.drop(columns = ["index"])

#Gets the length of each of the lyrics
df_3["length"] = df_3["lyrics_processed"].map(lambda x: len(x))
print(len(df_3))
df_3 = df_3[df_3["length"] < 5000]
df_3 = df_3[df_3["length"] > 400]
print(len(df_3))

#Split text up so that words can be counted
df_3["text_split"] = df_3["lyrics_processed"].map(lambda x: x.split(" "))

#Make list so that they may be counted
words_list = df_3["text_split"].tolist()
flat_list = [item for sublist in words_list for item in sublist]

#Gets counts of words
word_counts = Counter(flat_list)

#Returns 100 most common words from the documents used to determine stops
word_counts.most_common(100)
```
# Model Building
So now that the data is cleaned, I needed to determine a model that could determine the topic of a song without a label of what the song was about since that data was unavaialbe from our source. I decided to utilize Latent Dirichlet Allocation (LDA) as the model since it works quickly and does not need labeled data to generate the topics of song. Below is the code utilized the generate the LDA model and output.

```python
#Creates the tokens for the words
def sent_to_words(sentences):
    for sentence in sentences:
        #Use yield instead of return to save memory in the process
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#Gets all lyrics data to list
data = df_3.lyrics_processed.values.tolist()
data_words = list(sent_to_words(data))

# Remove Stop Words
data_words_nostops = [[word for word in song if word not in all_stopwords_gensim] for song in data_words]

# Create bigram and trigram model
# Build the bigram and trigram models
bigram = Phrases(data_words, min_count=4, threshold=40)
trigram = Phrases(bigram[data_words], threshold=40)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_words_bigrams = [bigram_mod[song] for song in data_words_nostops]
data_words_trigrams = [trigram_mod[bigram_mod[song]] for song in data_words_nostops]

#Free up some memory
import gc
import pandas as pd

del(df)
gc.collect()
df=pd.DataFrame()
del(df_2)
gc.collect()
del(data_words)

#Removing allowed_postags for now may need to re-add back in depending on smaller results
#Decided to add posttags as they produce better results filtering out unwanted words
def lemmatization(song_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    songs_out = []
    for sent in song_list:
        doc = nlp(" ".join(sent)) 
        songs_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return songs_out

#Load in spacy that will allow for lemmatization to occur
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

del(data_words_nostops)

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

#Creates another refrence for lemmatized data
texts = data_lemmatized

#Creates the corpus
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
start_time = time.time()
lda_model = gensim.models.ldamulticore.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=175, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=3000,
                                           passes=10,
                                           alpha=0.2,
                                           per_word_topics=True,
                                           eta = 0.21)
end_time = time.time()
print("Run time: " + str(end_time-start_time))

# Compute Perplexity
start_time = time.time()
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
end_time = time.time()
print("Run time: " + str(end_time-start_time))

#Generate the CSV Output
#Code is repeated 4 parts to break up the data and save memory
topics_words = lda_model.show_topics(formatted=False, num_words= 5, num_topics = 175)

df_topics = pd.DataFrame(columns = ["topic","word1","word2","word3","word4","word5"])
for i in range(175):
    df_topics = df_topics.append({"topic": i, "word1": topics_words[i][1][0][0],"word2": topics_words[i][1][1][0],"word3": topics_words[i][1][2][0],"word4": topics_words[i][1][3][0],"word5": topics_words[i][1][4][0]}, ignore_index=True)
df_topics.astype({"topic":"int64"}).dtypes


top_topics_vecs = []
counter = 0


for i in range(0,20000):
    all_topics = (
        lda_model.get_document_topics(corpus[i],
                                      minimum_probability=0.0)
    )
    all_topic_vec = [all_topics[i][1] for i in range(175)]
    top_topics_vecs.append(all_topic_vec)
    counter += 1
    if counter % 1000 == 0:
      print(counter)

max_topic = []
for i in top_topics_vecs:
    max_topic.append(i.index(max(i)))

temp_df = df_3[0:20000]
temp_df["topic"] = max_topic
temp_df = temp_df.join(df_topics, on = "topic", lsuffix="",rsuffix="_righttopic")

temp_df.to_csv('Model_1_output_175.csv')

top_topics_vecs = []
counter = 0


for i in range(20000,40000):
    all_topics = (
        lda_model.get_document_topics(corpus[i],
                                      minimum_probability=0.0)
    )
    all_topic_vec = [all_topics[i][1] for i in range(175)]
    top_topics_vecs.append(all_topic_vec)
    counter += 1
    if counter % 1000 == 0:
      print(counter)

max_topic = []
for i in top_topics_vecs:
    max_topic.append(i.index(max(i)))

temp_df = df_3[20000:40000]
temp_df["topic"] = max_topic
temp_df = temp_df.join(df_topics, on = "topic", lsuffix="",rsuffix="_righttopic")

temp_df.to_csv('Model_2_output_175.csv')

top_topics_vecs = []
counter = 0


for i in range(40000,60000):
    all_topics = (
        lda_model.get_document_topics(corpus[i],
                                      minimum_probability=0.0)
    )
    all_topic_vec = [all_topics[i][1] for i in range(175)]
    top_topics_vecs.append(all_topic_vec)
    counter += 1
    if counter % 1000 == 0:
      print(counter)

max_topic = []
for i in top_topics_vecs:
    max_topic.append(i.index(max(i)))

temp_df = df_3[40000:60000]
temp_df["topic"] = max_topic
temp_df = temp_df.join(df_topics, on = "topic", lsuffix="",rsuffix="_righttopic")

temp_df.to_csv('Model_3_output_175.csv')

top_topics_vecs = []
counter = 0


for i in range(60000,len(corpus)):
    all_topics = (
        lda_model.get_document_topics(corpus[i],
                                      minimum_probability=0.0)
    )
    all_topic_vec = [all_topics[i][1] for i in range(175)]
    top_topics_vecs.append(all_topic_vec)
    counter += 1
    if counter % 1000 == 0:
      print(counter)

max_topic = []
for i in top_topics_vecs:
    max_topic.append(i.index(max(i)))

temp_df = df_3[60000:len(corpus)]
temp_df["topic"] = max_topic
temp_df = temp_df.join(df_topics, on = "topic", lsuffix="",rsuffix="_righttopic")

temp_df.to_csv('Model_4_output_175.csv')
```
The code above generates the LDA model utlized to generate the topics for each song and outputs it as a csv file. There were numerous hyperparmeters that needed to be tuned to generate the best model. This was completed by changing the parameters of the number of topics, alpha, and eta and running the model numerous time to generate the best coherence score. With the output of topics, our team then created an interactive visualization so that users can find their own interesting trends! The visualization can be accessed here: https://public.tableau.com/app/profile/taegeun.ohe/viz/Visualization_16384078765610/TopTopicsinGenres#2

The visualization allows the user to break down the most popular topics for each genre, see key words that create a topic, and set timeframes to generate their own insights and see interesting trends! This wraps up the work done for this project!
