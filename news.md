```python
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import nltk
import re
from tqdm import tqdm
import plotly.express as px
from decimal import *
```


```python
true_data_original = pd.read_csv("./True.csv")
fake_data_original = pd.read_csv("./Fake.csv")
```


```python
true_data_original
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>As U.S. budget fight looms, Republicans flip t...</td>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U.S. military to accept transgender recruits o...</td>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FBI Russia probe helped by Australian diplomat...</td>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>politicsNews</td>
      <td>December 30, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump wants Postal Service to charge 'much mor...</td>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21412</th>
      <td>'Fully committed' NATO backs new U.S. approach...</td>
      <td>BRUSSELS (Reuters) - NATO allies on Tuesday we...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
    </tr>
    <tr>
      <th>21413</th>
      <td>LexisNexis withdrew two products from Chinese ...</td>
      <td>LONDON (Reuters) - LexisNexis, a provider of l...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
    </tr>
    <tr>
      <th>21414</th>
      <td>Minsk cultural hub becomes haven from authorities</td>
      <td>MINSK (Reuters) - In the shadow of disused Sov...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
    </tr>
    <tr>
      <th>21415</th>
      <td>Vatican upbeat on possibility of Pope Francis ...</td>
      <td>MOSCOW (Reuters) - Vatican Secretary of State ...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
    </tr>
    <tr>
      <th>21416</th>
      <td>Indonesia to buy $1.14 billion worth of Russia...</td>
      <td>JAKARTA (Reuters) - Indonesia will buy 11 Sukh...</td>
      <td>worldnews</td>
      <td>August 22, 2017</td>
    </tr>
  </tbody>
</table>
<p>21417 rows × 4 columns</p>
</div>




```python
fake_data_original
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Donald Trump Sends Out Embarrassing New Year’...</td>
      <td>Donald Trump just couldn t wish all Americans ...</td>
      <td>News</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Drunk Bragging Trump Staffer Started Russian ...</td>
      <td>House Intelligence Committee Chairman Devin Nu...</td>
      <td>News</td>
      <td>December 31, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sheriff David Clarke Becomes An Internet Joke...</td>
      <td>On Friday, it was revealed that former Milwauk...</td>
      <td>News</td>
      <td>December 30, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>
      <td>On Christmas day, Donald Trump announced that ...</td>
      <td>News</td>
      <td>December 29, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pope Francis Just Called Out Donald Trump Dur...</td>
      <td>Pope Francis used his annual Christmas Day mes...</td>
      <td>News</td>
      <td>December 25, 2017</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23476</th>
      <td>McPain: John McCain Furious That Iran Treated ...</td>
      <td>21st Century Wire says As 21WIRE reported earl...</td>
      <td>Middle-east</td>
      <td>January 16, 2016</td>
    </tr>
    <tr>
      <th>23477</th>
      <td>JUSTICE? Yahoo Settles E-mail Privacy Class-ac...</td>
      <td>21st Century Wire says It s a familiar theme. ...</td>
      <td>Middle-east</td>
      <td>January 16, 2016</td>
    </tr>
    <tr>
      <th>23478</th>
      <td>Sunnistan: US and Allied ‘Safe Zone’ Plan to T...</td>
      <td>Patrick Henningsen  21st Century WireRemember ...</td>
      <td>Middle-east</td>
      <td>January 15, 2016</td>
    </tr>
    <tr>
      <th>23479</th>
      <td>How to Blow $700 Million: Al Jazeera America F...</td>
      <td>21st Century Wire says Al Jazeera America will...</td>
      <td>Middle-east</td>
      <td>January 14, 2016</td>
    </tr>
    <tr>
      <th>23480</th>
      <td>10 U.S. Navy Sailors Held by Iranian Military ...</td>
      <td>21st Century Wire says As 21WIRE predicted in ...</td>
      <td>Middle-east</td>
      <td>January 12, 2016</td>
    </tr>
  </tbody>
</table>
<p>23481 rows × 4 columns</p>
</div>




```python
test_count = 4000

fake_data = fake_data_original[0:fake_data_original.shape[0] - test_count]
true_data = true_data_original[0:true_data_original.shape[0] - test_count]

print("Amount of true news: {}".format(true_data.shape[0]))
print("Amount of fake news: {}\n".format(fake_data.shape[0]))

data = fake_data.append(true_data)

print("Amount of all news: {}\n".format(data.shape[0]))

fake_data_test = fake_data_original[fake_data_original.shape[0] - test_count:]
true_data_test = true_data_original[true_data_original.shape[0] - test_count:]
fake_data_test.reset_index(drop=True, inplace=True)
true_data_test.reset_index(drop=True, inplace=True)

print("Amount of true test news: {}".format(true_data_test.shape[0]))
print("Amount of fake test news: {}".format(fake_data_test.shape[0]))
```

    Amount of true news: 17417
    Amount of fake news: 19481
    
    Amount of all news: 36898
    
    Amount of true test news: 4000
    Amount of fake test news: 4000
    


```python
nltk.download("stopwords")
tqdm.pandas()
pd.set_option('mode.chained_assignment', None)

def preprocess(df):
    stopwords = nltk.corpus.stopwords.words('english')
    df['text_pre'] = df['text']
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : x.lower())
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : x.split(" "))
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : [item for item in x if item not in stopwords])
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : " ".join(x))
    #df['text_pre'] = df['text_pre'].str.replace('@[^\s]+', "")
    df['text_pre'] = df['text_pre'].str.replace('@([A-Za-z0-9_]+)', "twittername")
    df['text_pre'] = df['text_pre'].str.replace('https?:\/\/.*[\r\n]*', '')
    df['text_pre'] = df['text_pre'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df['text_pre'] = df['text_pre'].str.replace('\d+', '')
    df['text_pre'] = df['text_pre'].str.replace('[^\w\s]', '')
    return df
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\SSJSR\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    


```python
fake_data = preprocess(fake_data)
true_data = preprocess(true_data)
data = preprocess(data)
```

    100%|██████████| 19481/19481 [00:00<00:00, 219471.49it/s]
    100%|██████████| 19481/19481 [00:00<00:00, 22021.52it/s]
    100%|██████████| 19481/19481 [00:16<00:00, 1212.26it/s]
    100%|██████████| 19481/19481 [00:00<00:00, 110397.15it/s]
    100%|██████████| 17417/17417 [00:00<00:00, 65400.06it/s]
    100%|██████████| 17417/17417 [00:00<00:00, 25944.64it/s]
    100%|██████████| 17417/17417 [00:14<00:00, 1184.66it/s]
    100%|██████████| 17417/17417 [00:00<00:00, 69302.84it/s]
    100%|██████████| 36898/36898 [00:00<00:00, 94379.44it/s]
    100%|██████████| 36898/36898 [00:01<00:00, 21585.00it/s]
    100%|██████████| 36898/36898 [00:31<00:00, 1188.45it/s]
    100%|██████████| 36898/36898 [00:00<00:00, 96095.15it/s]
    


```python
fake_data.loc[0]["text"]
```




    'Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year. 2018 will be a great year for America!  Donald J. Trump (@realDonaldTrump) December 31, 2017Trump s tweet went down about as welll as you d expect.What kind of president sends a New Year s greeting like this despicable, petty, infantile gibberish? Only Trump! His lack of decency won t even allow him to rise above the gutter long enough to wish the American citizens a happy new year!  Bishop Talbert Swan (@TalbertSwan) December 31, 2017no one likes you  Calvin (@calvinstowell) December 31, 2017Your impeachment would make 2018 a great year for America, but I ll also accept regaining control of Congress.  Miranda Yaver (@mirandayaver) December 31, 2017Do you hear yourself talk? When you have to include that many people that hate you you have to wonder? Why do the they all hate me?  Alan Sandoval (@AlanSandoval13) December 31, 2017Who uses the word Haters in a New Years wish??  Marlene (@marlene399) December 31, 2017You can t just say happy new year?  Koren pollitt (@Korencarpenter) December 31, 2017Here s Trump s New Year s Eve tweet from 2016.Happy New Year to all, including to my many enemies and those who have fought me and lost so badly they just don t know what to do. Love!  Donald J. Trump (@realDonaldTrump) December 31, 2016This is nothing new for Trump. He s been doing this for years.Trump has directed messages to his  enemies  and  haters  for New Year s, Easter, Thanksgiving, and the anniversary of 9/11. pic.twitter.com/4FPAe2KypA  Daniel Dale (@ddale8) December 31, 2017Trump s holiday tweets are clearly not presidential.How long did he work at Hallmark before becoming President?  Steven Goodine (@SGoodine) December 31, 2017He s always been like this . . . the only difference is that in the last few years, his filter has been breaking down.  Roy Schulze (@thbthttt) December 31, 2017Who, apart from a teenager uses the term haters?  Wendy (@WendyWhistles) December 31, 2017he s a fucking 5 year old  Who Knows (@rainyday80) December 31, 2017So, to all the people who voted for this a hole thinking he would change once he got into power, you were wrong! 70-year-old men don t change and now he s a year older.Photo by Andrew Burton/Getty Images.'




```python
fake_data.loc[0]["text_pre"]
```




    'donald trump wish americans happy new year leave that instead give shout enemies haters  dishonest fake news media  former reality show star one job it country rapidly grows stronger smarter want wish friends supporters enemies haters even dishonest fake news media happy healthy new year  president angry pants tweeted   great year america country rapidly grows stronger smarter want wish friends supporters enemies haters even dishonest fake news media happy healthy new year  great year america  donald j trump twittername december  trump tweet went welll expectwhat kind president sends new year greeting like despicable petty infantile gibberish trump lack decency even allow rise gutter long enough wish american citizens happy new year  bishop talbert swan twittername december  no one likes  calvin twittername december  your impeachment would make  great year america also accept regaining control congress  miranda yaver twittername december  do hear talk include many people hate wonder hate me  alan sandoval twittername december  who uses word haters new years wish  marlene twittername december  you say happy new year  koren pollitt twittername december  here trump new year eve tweet happy new year all including many enemies fought lost badly know do love  donald j trump twittername december  this nothing new trump yearstrump directed messages  enemies   haters  new year s easter thanksgiving anniversary  pictwittercomfpaekypa  daniel dale twittername december  trump holiday tweets clearly presidentialhow long work hallmark becoming president  steven goodine twittername december  he always like    difference last years filter breaking down  roy schulze twittername december  who apart teenager uses term haters  wendy twittername december  he fucking  year old  knows twittername december  so people voted hole thinking would change got power wrong yearold men change year olderphoto andrew burtongetty images'




```python
fake_data['is_fake'] = 1
true_data['is_fake'] = 0
```


```python
def text_size(df):
    sizes = []
    for text in tqdm(df['text']):
        len_ = len(text.split())
        sizes.append(len_)
    return np.array(sizes)

def text_size_preprocess(df):
    sizes = []
    for text in tqdm(df['text_pre']):
        len_ = len(text.split())
        sizes.append(len_)
    return np.array(sizes)
```


```python
fake_size = text_size(fake_data)
true_size = text_size(true_data)
fake_data['len'] = fake_size
true_data['len'] = true_size
```

    100%|██████████| 19481/19481 [00:00<00:00, 36239.34it/s]
    100%|██████████| 17417/17417 [00:00<00:00, 37394.93it/s]
    


```python
true_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>text_pre</th>
      <th>is_fake</th>
      <th>len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>As U.S. budget fight looms, Republicans flip t...</td>
      <td>WASHINGTON (Reuters) - The head of a conservat...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
      <td>washington reuters  head conservative republic...</td>
      <td>0</td>
      <td>749</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U.S. military to accept transgender recruits o...</td>
      <td>WASHINGTON (Reuters) - Transgender people will...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
      <td>washington reuters  transgender people allowed...</td>
      <td>0</td>
      <td>624</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>
      <td>WASHINGTON (Reuters) - The special counsel inv...</td>
      <td>politicsNews</td>
      <td>December 31, 2017</td>
      <td>washington reuters  special counsel investigat...</td>
      <td>0</td>
      <td>457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FBI Russia probe helped by Australian diplomat...</td>
      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>
      <td>politicsNews</td>
      <td>December 30, 2017</td>
      <td>washington reuters  trump campaign adviser geo...</td>
      <td>0</td>
      <td>376</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trump wants Postal Service to charge 'much mor...</td>
      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>
      <td>politicsNews</td>
      <td>December 29, 2017</td>
      <td>seattlewashington reuters  president donald tr...</td>
      <td>0</td>
      <td>852</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.box(pd.concat([fake_data, true_data]), y='len', x='is_fake', title="Distribution of news' length", width=1280, height=720)
fig.show()
```




```python
fake_size_pre = text_size_preprocess(fake_data)
true_size_pre = text_size_preprocess(true_data)
fake_data["len_pre"] = fake_size_pre
true_data["len_pre"] = true_size_pre
```

    100%|██████████| 19481/19481 [00:00<00:00, 63419.20it/s]
    100%|██████████| 17417/17417 [00:00<00:00, 58602.18it/s]
    


```python
fig = px.box(pd.concat([fake_data, true_data]), y='len_pre', x='is_fake', title="Distribution of news' length (preprocessed)", width=1280, height=720)
fig.show()
```




```python
def unique_words(df):
    unique_words = set()
    for words in tqdm(df["text"]):
        splited = words.split()
        for word in splited:
            unique_words.add(word)
    return unique_words

def unique_words_preprocess(df):
    unique_words = set()
    for words in tqdm(df["text_pre"]):
        splited = words.split()
        for word in splited:
            unique_words.add(word)
    return unique_words
```


```python
true_unique_words = unique_words(true_data)
fake_unique_words = unique_words(fake_data)
```

    100%|██████████| 17417/17417 [00:01<00:00, 9922.49it/s]
    100%|██████████| 19481/19481 [00:01<00:00, 12691.38it/s]
    


```python
fig = px.bar(y=[len(true_unique_words), len(fake_unique_words)], x=["true", "fake"], title="Unique words", width=1280, height=720)
fig.show()
```




```python
true_unique_words_pre = unique_words_preprocess(true_data)
fake_unique_words_pre = unique_words_preprocess(fake_data)
```

    100%|██████████| 17417/17417 [00:00<00:00, 18899.48it/s]
    100%|██████████| 19481/19481 [00:00<00:00, 22421.59it/s]
    


```python
fig = px.bar(y=[len(true_unique_words_pre), len(fake_unique_words_pre)], x=["true", "fake"], title="Unique words (preprocessed)", width=1280, height=720)
fig.show()
```




```python
word_distribution = {}
total_fake_words = 0
total_true_words = 0

for idx in tqdm(fake_data.index):
    words = fake_data.loc[idx]["text_pre"].split()
    total_fake_words += len(words)
    for word in words:
        if word not in word_distribution:
            word_distribution[word] = {"true": 1, "fake": 1}
        else:
            word_distribution[word]["fake"] += 1

for idx in tqdm(true_data.index):
    words = true_data.loc[idx]["text_pre"].split()
    total_true_words += len(words)
    for word in words:
        if word not in word_distribution:
            word_distribution[word] = {"true": 1, "fake": 1}
        else:
            word_distribution[word]["true"] += 1
```

    100%|██████████| 19481/19481 [00:05<00:00, 3569.01it/s]
    100%|██████████| 17417/17417 [00:04<00:00, 3484.24it/s]
    


```python
print(word_distribution["turkey"])
```

    {'true': 1486, 'fake': 166}
    


```python
print(word_distribution["reuters"])
```

    {'true': 22979, 'fake': 244}
    


```python
true_words = [word_distribution[k]["true"] for k in word_distribution.keys()]
fake_words = [word_distribution[k]["fake"] for k in word_distribution.keys()]
names = [k for k in word_distribution.keys()]
```


```python
frm = pd.DataFrame(zip(true_words, fake_words), columns=["true_freq", "false_freq"])
fig = px.scatter(frm, x="true_freq", y="false_freq", hover_name=names, width=1280, height=720, title="Frequency of words")
fig.show()
```




```python
fig = px.scatter(frm, x="true_freq", y="false_freq", hover_name=names, log_x=True, log_y=True, width=1280, height=720, title="Frequency of words (log scaled)")
fig.show()
```




```python
fake_word_counts = {k: word_distribution[k]['fake'] for k in word_distribution.keys()}
fake_keys = list(fake_word_counts.keys())
fake_values = list(fake_word_counts.values())
print(*list(zip(fake_keys[0: 10], fake_values[0: 10])), sep="\n")
```

    ('donald', 14718)
    ('trump', 62943)
    ('wish', 456)
    ('americans', 4794)
    ('happy', 709)
    ('new', 10008)
    ('year', 5203)
    ('leave', 1249)
    ('that', 4071)
    ('instead', 2269)
    


```python
fake_word_counts = {k: v for k, v in sorted(fake_word_counts.items(), key=lambda item: item[1], reverse=True)}
fake_keys = list(fake_word_counts.keys())
fake_values = list(fake_word_counts.values())
print(*list(zip(fake_keys[0: 10], fake_values[0: 10])), sep="\n")
```

    ('trump', 62943)
    ('said', 23921)
    ('president', 20774)
    ('people', 20559)
    ('would', 18019)
    ('one', 17184)
    ('twittername', 16004)
    ('donald', 14718)
    ('obama', 13952)
    ('like', 13451)
    


```python
fake_frequency = pd.DataFrame(zip(fake_keys[0: 50], fake_values[0: 50]), columns=["word", "frequency"])
fig = px.bar(fake_frequency, x="word", y="frequency", title="Fake news frequency words", width=1280, height=720)
fig.show()
```




```python
true_word_counts = {k: word_distribution[k]['true'] for k in word_distribution.keys()}
true_word_counts = {k: v for k, v in sorted(true_word_counts.items(), key=lambda item: item[1], reverse=True)}
true_keys = list(true_word_counts.keys())
true_values = list(true_word_counts.values())
```


```python
true_frequency = pd.DataFrame(zip(true_keys[0: 50], true_values[0: 50]), columns=["word", "frequency"])
fig = px.bar(true_frequency, x="word", y="frequency", title="True news frequency words", width=1280, height=720)
fig.show()
```




```python
text_general = " ".join(dta for dta in data['text_pre'])
print("There are {} words.".format(len(text_general.split())))
```

    There are 8050501 words.
    


```python
general_color_mask = np.array(Image.open("./masks/color_mask9.png"))
general_image_colors = ImageColorGenerator(general_color_mask)
```


```python
wordcloud_general = WordCloud(background_color="white", max_words=1000, color_func=general_image_colors, min_font_size=7, width=1280, height=720).generate(text_general)
```


```python
# General Wordcloud
plt.rcParams["figure.figsize"] = (20,20)
plt.imshow(wordcloud_general, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![svg](output_36_0.svg)



```python
fake_text = " ".join(fakes for fakes in fake_data['text_pre'])
print("There are {} words in fake news.".format(len(fake_text.split())))
```

    There are 3991523 words in fake news.
    


```python
fake_color_mask = np.array(Image.open("./masks/color_mask10.png"))
fake_image_colors = ImageColorGenerator(fake_color_mask)
```


```python
wordcloud_fake = WordCloud(background_color="white", max_words=1000, min_font_size=7, color_func=fake_image_colors, width=1280, height=720).generate(fake_text)
```


```python
# Fake Wordcloud
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![svg](output_40_0.svg)



```python
true_text = " ".join(trues for trues in true_data['text_pre'])
print("There are {} words in true news.".format(len(true_text.split())))
```

    There are 4058978 words in true news.
    


```python
true_color_mask = np.array(Image.open("./masks/color_mask5.png"))
true_image_colors = ImageColorGenerator(true_color_mask)
```


```python
wordcloud_true = WordCloud(background_color="white", max_words=1000, color_func=true_image_colors, min_font_size=7, width=1280, height=720).generate(true_text)
```


```python
# True Wordcloud
plt.imshow(wordcloud_true, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![svg](output_44_0.svg)



```python
trump_mask = np.array(Image.open("./masks/trump.png"))
trump_mask
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)




```python
transformed_trump_mask = np.ndarray([trump_mask.shape[0], trump_mask.shape[1]], np.int32)
for i in tqdm(range(len(trump_mask))):
    transformed_trump_mask[i] = list(map(lambda x: 255 if x == 0 else x, trump_mask[i]))
transformed_trump_mask
```

    100%|██████████| 730/730 [00:01<00:00, 400.82it/s]
    




    array([[255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           ...,
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255]])




```python
wordcloud_fake_trump = WordCloud(background_color="white", mask=transformed_trump_mask, max_words=2000, contour_color="black", contour_width=2, color_func=fake_image_colors, width=1280, height=720).generate(fake_text)
```


```python
# Masked Fake Wordcloud
plt.imshow(wordcloud_fake_trump, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![svg](output_48_0.svg)



```python
truth_mask = np.array(Image.open("./masks/truth.png"))
truth_mask
```




    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [254, 254, 254],
            [254, 254, 254],
            [254, 254, 254]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [254, 254, 254],
            [254, 254, 254],
            [253, 253, 253]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [253, 253, 253],
            [253, 253, 253],
            [252, 252, 252]],
    
           [[ 51,  51,  51],
            [ 53,  53,  53],
            [ 53,  53,  53],
            ...,
            [ 52,  52,  52],
            [ 51,  51,  51],
            [ 52,  52,  52]],
    
           [[ 51,  51,  51],
            [ 50,  50,  50],
            [ 54,  54,  54],
            ...,
            [ 52,  52,  52],
            [ 51,  51,  51],
            [ 54,  54,  54]]], dtype=uint8)




```python
red, green, blue = truth_mask[:, :, 0], truth_mask[:, :, 1], truth_mask[:, :, 2]
mask = (red < 128) & (green < 128) & (blue < 128)
mask2 = (red >= 128) & (green >= 128) & (blue >= 128)
truth_mask[:, :, :3][mask] = [0, 0, 0]
truth_mask[:, :, :3][mask2] = [255, 255, 255]
truth_mask
```




    array([[[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           ...,
    
           [[255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            ...,
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255]],
    
           [[  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            ...,
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]],
    
           [[  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            ...,
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]]], dtype=uint8)




```python
wordcloud_true_truth = WordCloud(background_color="white", max_words=5000, mask=truth_mask, color_func=true_image_colors, width=1280, height=720).generate(true_text)
```


```python
# Masked True Wordcloud
plt.imshow(wordcloud_true_truth, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![svg](output_52_0.svg)



```python
print("Total true words: {}".format(total_true_words))
print("Total fake words: {}\n".format(total_fake_words))

p_true = total_true_words / (total_true_words + total_fake_words)
p_fake = total_fake_words / (total_true_words + total_fake_words)

print("General probability of true words: {}"
"\nGeneral probability of fake words: {}".format(p_true, p_fake))
```

    Total true words: 4058978
    Total fake words: 3991523
    
    General probability of true words: 0.5041894908155405
    General probability of fake words: 0.49581050918445946
    


```python
probabilities = {}
for word in tqdm(word_distribution):
    p_word_true = word_distribution[word]["true"] / total_true_words
    p_word_fake = word_distribution[word]["fake"] / total_fake_words
    probabilities[word] = {"true": p_word_true, "fake": p_word_fake}
```

    100%|██████████| 173221/173221 [00:00<00:00, 542763.05it/s]
    


```python
word = "trump"
print("Raw probabilities: {}\n".format(probabilities[word]))

print("The word '{}' is in {:.2f}% true news.".format(word, probabilities[word]["true"] * 100 / (probabilities[word]["true"] + probabilities[word]["fake"])))
print("The word '{}' is in {:.2f}% fake news.".format(word, probabilities[word]["fake"] * 100 / (probabilities[word]["true"] + probabilities[word]["fake"])))
```

    Raw probabilities: {'true': 0.010015575349262795, 'fake': 0.015769168811002716}
    
    The word 'trump' is in 38.84% true news.
    The word 'trump' is in 61.16% fake news.
    


```python
true_data_test = preprocess(true_data_test)
fake_data_test = preprocess(fake_data_test)
```

    100%|██████████| 4000/4000 [00:00<00:00, 286491.28it/s]
    100%|██████████| 4000/4000 [00:00<00:00, 27283.62it/s]
    100%|██████████| 4000/4000 [00:03<00:00, 1303.82it/s]
    100%|██████████| 4000/4000 [00:00<00:00, 108401.66it/s]
    100%|██████████| 4000/4000 [00:00<00:00, 285759.33it/s]
    100%|██████████| 4000/4000 [00:00<00:00, 24911.12it/s]
    100%|██████████| 4000/4000 [00:04<00:00, 918.39it/s]
    100%|██████████| 4000/4000 [00:00<00:00, 83556.03it/s]
    


```python
X = fake_data_test.loc[0]["text_pre"]
X = [i for i in X.split()]
print(X)
```

    ['pretty', 'scary', 'stuff', 'federal', 'government', 'agency', 'run', 'obama', 'crony', 'attempting', 'penetrate', 'firewall', 'state', 'agency', 'tasked', 'overseeing', 'elections', 'conceivable', 'reason', 'could', 'obama', 'dhs', 'hacking', 'georgia', 'sec', 'state', 'office', 'electiongeorgia', 'secretary', 'state', 'claimed', 'department', 'homeland', 'security', 'tried', 'breach', 'office', 'firewall', 'issued', 'letter', 'homeland', 'security', 'secretary', 'jeh', 'johnson', 'asking', 'explanationbrian', 'kemp', 'issued', 'letter', 'johnson', 'thursday', 'state', 'thirdparty', 'cybersecurity', 'provider', 'detected', 'ip', 'address', 'agency', 'southwest', 'dc', 'office', 'trying', 'penetrate', 'state', 'firewall', 'according', 'letter', 'attempt', 'unsuccessfulthe', 'attempt', 'took', 'place', 'nov', 'days', 'presidential', 'election', 'office', 'georgia', 'secretary', 'state', 'responsible', 'overseeing', 'state', 'elections', 'time', 'office', 'agreed', 'permitted', 'dhs', 'conduct', 'penetration', 'testing', 'security', 'scans', 'network', 'kemp', 'wrote', 'letter', 'also', 'sent', 'state', 'federal', 'representatives', 'senators', 'moreover', 'department', 'contacted', 'office', 'since', 'unsuccessful', 'incident', 'alert', 'us', 'security', 'event', 'would', 'require', 'testing', 'scanning', 'network', 'especially', 'odd', 'concerning', 'since', 'serve', 'election', 'cyber', 'security', 'working', 'group', 'office', 'created', 'department', 'homeland', 'security', 'received', 'secretary', 'kemp', 'letter', 'dhs', 'spokesperson', 'told', 'cyberscoop', 'looking', 'matter', 'dhs', 'takes', 'trust', 'public', 'private', 'sector', 'partners', 'seriously', 'respond', 'secretary', 'kemp', 'directly', 'georgia', 'one', 'two', 'states', 'refused', 'cyberhygiene', 'support', 'penetration', 'testing', 'dhs', 'lead', 'presidential', 'election', 'department', 'made', 'significant', 'push', 'hackers', 'spent', 'months', 'exposing', 'democratic', 'national', 'committee', 'internal', 'communications', 'datadavid', 'dove', 'kemp', 'chief', 'staff', 'told', 'cyberscoop', 'georgia', 'secretary', 'state', 'office', 'got', 'lot', 'grief', 'refusing', 'help', 'dhs', 'basically', 'said', 'need', 'dhs', 'help', 'contract', 'thirdparty', 'provider', 'dove', 'saidthe', 'office', 'georgia', 'secretary', 'state', 'would', 'reveal', 'provider', 'is', 'saying', 'company', 'analyzes', 'billion', 'events', 'day', 'globally', 'across', 'customer', 'base', 'includes', 'many', 'fortune', 'companies', 'johnson', 'announced', 'shortly', 'election', 'dhs', 'found', 'evidence', 'attack', 'election', 'dayin', 'months', 'weeks', 'leading', 'election', 'fake', 'news', 'sources', 'like', 'washington', 'post', 'everything', 'power', 'convince', 'americans', 'georgia', 'play', 'hillary', 'attempt', 'took', 'place', 'nov', 'days', 'presidential', 'election', 'office', 'georgia', 'secretary', 'state', 'responsible', 'overseeing', 'state', 'elections', 'time', 'office', 'agreed', 'permitted', 'dhs', 'conduct', 'penetration', 'testing', 'security', 'scans', 'network', 'kemp', 'wrote', 'letter', 'also', 'sent', 'state', 'federal', 'representatives', 'senators', 'moreover', 'department', 'contacted', 'office', 'since', 'unsuccessful', 'incident', 'alert', 'us', 'security', 'event', 'would', 'require', 'testing', 'scanning', 'network', 'especially', 'odd', 'concerning', 'since', 'serve', 'election', 'cyber', 'security', 'working', 'group', 'office', 'created', 'department', 'homeland', 'security', 'received', 'secretary', 'kemp', 'letter', 'dhs', 'spokesperson', 'told', 'cyberscoop', 'looking', 'matter', 'dhs', 'takes', 'trust', 'public', 'private', 'sector', 'partners', 'seriously', 'respond', 'secretary', 'kemp', 'directly', 'georgia', 'one', 'two', 'states', 'refused', 'cyberhygiene', 'support', 'penetration', 'testing', 'dhs', 'leadup', 'presidential', 'election', 'department', 'made', 'significant', 'push', 'hackers', 'spent', 'months', 'exposing', 'democratic', 'national', 'committee', 'internal', 'communications', 'datain', 'interview', 'politico', 'kemp', 'intimated', 'federal', 'government', 'hacking', 'fears', 'overblown', 'saying', 'think', 'whole', 'system', 'verge', 'disaster', 'russian', 'going', 'tap', 'voting', 'system', 'david', 'dove', 'kemp', 'chief', 'staff', 'told', 'cyberscoop', 'georgia', 'secretary', 'state', 'office', 'got', 'lot', 'grief', 'refusing', 'help', 'dhs', 'basically', 'said', 'need', 'dhs', 'help', 'contract', 'thirdparty', 'provider', 'dove', 'saidthe', 'office', 'georgia', 'secretary', 'state', 'would', 'reveal', 'provider', 'is', 'saying', 'company', 'analyzes', 'billion', 'events', 'day', 'globally', 'across', 'customer', 'base', 'includes', 'many', 'fortune', 'companies', 'majority', 'states', 'worked', 'dhs', 'help', 'protecting', 'election', 'systems', 'hacks', 'cybersecurity', 'experts', 'oddsas', 'portions', 'country', 'would', 'targeted', 'election', 'day', 'attacksjohnson', 'announced', 'shortly', 'election', 'dhs', 'found', 'evidence', 'attack', 'election', 'day', 'analyzes', 'billion', 'events', 'day', 'globally', 'across', 'customer', 'base', 'includes', 'many', 'fortune', 'companies', 'majority', 'states', 'worked', 'dhs', 'help', 'protecting', 'election', 'systems', 'hacks', 'cybersecurity', 'experts', 'oddsas', 'portions', 'country', 'would', 'targeted', 'election', 'day', 'attacksjohnson', 'announced', 'shortly', 'election', 'dhs', 'found', 'evidence', 'attack', 'election', 'daythe', 'manipulation', 'minds', 'americans', 'witnessed', 'leftist', 'media', 'leading', 'to', 'election', 'unprecedented', 'leftist', 'news', 'sources', 'like', 'washington', 'post', 'tried', 'convince', 'voters', 'hillary', 'shot', 'winning', 'solidly', 'red', 'state', 'georgia', 'new', 'atlanta', 'journalconstitution', 'poll', 'presidential', 'race', 'state', 'puts', 'hillary', 'clinton', 'slight', 'lead', 'donald', 'trump', 'insidethemarginoferror', 'stuff', 'mind', 'you', 'barely', 'it', 'also', 'break', 'state', 'race', 'realclearpolitics', 'polling', 'average', 'gives', 'trump', 'fourpoint', 'lead', 'though', 'handful', 'polls', 'includedgeorgia', 'shifting', 'bit', 'politically', 'recent', 'presidential', 'elections', 'becoming', 'slightly', 'less', 'republican', 'relative', 'rest', 'country', 'still', 'voting', 'republican', 'america', 'whole', 'barack', 'obama', 'four', 'points', 'mitt', 'romney', 'georgia', 'eight', 'making', 'state', 'points', 'republican', 'tick', 'less', 'with', 'many', 'forces', 'resources', 'used', 'donald', 'j', 'trump', 'absolutely', 'amazing', 'election']
    


```python
print(len(X))
```

    617
    


```python
X_true = Decimal(1.0)
X_fake = Decimal(1.0)

for word in X:
    if word in probabilities:
        X_true *= Decimal(probabilities[word]["true"])
        X_fake *= Decimal(probabilities[word]["fake"])
    else:
        print("'{}' word does not exist in probabilities!".format(word))
```


```python
print("P(X|true) = {}".format(Decimal(X_true)))
print("P(X|fake) = {}".format(Decimal(X_fake)))
# true_percent = (Decimal(X_true) * 100) / (Decimal(X_true) + Decimal(X_fake))
# false_percent = (Decimal(X_fake) * 100) / (Decimal(X_true) + Decimal(X_fake))
# print(true_percent)
# print(false_percent)
```

    P(X|true) = 8.600505711945510349506717628E-2297
    P(X|fake) = 2.672643257094507808344861314E-2289
    


```python
p_Xtrue_and_true = Decimal(X_true) * Decimal(p_true)
p_Xfake_and_fake = Decimal(X_fake) * Decimal(p_fake)

print("P(true|X) = {}".format(p_Xtrue_and_true))
print("P(fake|X) = {}".format(p_Xfake_and_fake))
```

    P(true|X) = 4.336284595661954860045150466E-2297
    P(fake|X) = 1.325124614168440104394958028E-2289
    


```python
if Decimal(p_Xtrue_and_true) > Decimal(p_Xfake_and_fake):
    print("TRUE")
else:
    print("FAKE")
```

    FAKE
    


```python
def decide(X, probabilities, p_true, p_fake):
    X_true = Decimal(1.0)
    X_fake = Decimal(1.0)
    for word in X.split():
        if word in probabilities:
            X_true *= Decimal(probabilities[word]["true"])
            X_fake *= Decimal(probabilities[word]["fake"])
        else:
            continue

    p_Xtrue_and_true = Decimal(X_true) * Decimal(p_true)
    p_Xfake_and_fake = Decimal(X_fake) * Decimal(p_fake)
    
    return p_Xtrue_and_true > p_Xfake_and_fake
```


```python
all_fake_tests = [True if not decide(fake_data_test.loc[X]["text_pre"], probabilities, p_true, p_fake) else False for X in tqdm(fake_data_test.index)]
```

    100%|██████████| 4000/4000 [00:04<00:00, 816.83it/s]
    


```python
fake_accuracy = all_fake_tests.count(True) / len(all_fake_tests)
print("Accuracy for fakes: {}".format(fake_accuracy))
```

    Accuracy for fakes: 0.8595
    


```python
all_true_tests = [True if decide(true_data_test.loc[X]["text_pre"], probabilities, p_true, p_fake) else False for X in tqdm(true_data_test.index)]
```

    100%|██████████| 4000/4000 [00:03<00:00, 1018.66it/s]
    


```python
true_accuracy = all_true_tests.count(True) / len(all_true_tests)
print("Accuracy for trues: {}".format(true_accuracy))
```

    Accuracy for trues: 0.98875
    


```python
fake_accuracy * true_accuracy
```




    0.8498306250000001




```python

```
