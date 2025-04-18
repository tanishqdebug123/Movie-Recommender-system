import pandas as pd
import sklearn as sk
import nltk


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

movies=pd.read_csv('dataset.csv')


def clean_text(text):
    if not isinstance(text,str):
        return ""
    
    text=text.lower()
    
    #remove punctuations
    text=re.sub(r'[^\w\s\d]','',text)
    
    #tokenise
    words_list=word_tokenize(text)
    
    stop_words=set(stopwords.words('english'))
    words_list=[word for word in words_list if word not in stop_words]
    
    #Lemmatise
    lemmatiser=WordNetLemmatizer()
    words_list=[lemmatiser.lemmatize(word) for word in words_list]
    text=' '.join(words_list)
    return text




# print(movies.head(10))
# print(movies.describe())
# print(movies.info())


#  Feature selection
movies["tags"]=movies["overview"]+movies["genre"]
new_data = movies[["id","title","tags"]].copy()

new_data['tags_clean']=new_data['tags'].apply(clean_text)

train_data,test_data=train_test_split(new_data,test_size=0.2,random_state=42)
cv=CountVectorizer(max_features=10000,stop_words='english')
vector=cv.fit_transform(new_data['tags_clean'].values.astype('U')).toarray()
similarity=cosine_similarity(vector)
# print(similarity)



def recommend_movies(movies):
    matches=new_data[new_data['title']==movies]
    if(matches.empty):
        print("Invalid movie name")
        return
    
    index=matches.index[0]    
    distance=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda vector:vector[1])
    for i in distance[0:10]:
        print(new_data.iloc[i[0]].title)
        
recommend_movies("Dilwale")



pickle.dump(new_data,open('movies_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

pickle.load(open('movies_list.pkl','rb'))