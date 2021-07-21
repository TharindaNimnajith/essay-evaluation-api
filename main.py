from fastapi import FastAPI
# import numpy as np
# import nltk
# import re
# from nltk.corpus import stopwords
# from language_tool_python import LanguageTool
from grammarbot import GrammarBotClient
# from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
# from keras.models import Sequential, load_model, model_from_config
# import keras.backend as K
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LinearRegression
# from gensim.models import Word2Vec

app = FastAPI()

# tool = LanguageTool('en-US')

# def grammar_spelling_mistakes(txt):
#     return tool.check(txt)

client = GrammarBotClient()

def grammar_spelling_mistakes(text):
    res = client.check(text)
    return res.raw_json.get('matches')

# import string

# def text_tokenizing(text, remove_stopwords):
#     punctuations = string.punctuation

#     text = re.sub("[^a-zA-Z]", " ", text)
#     words = text.lower().split()
    
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         word_tokens = [token for token in words if token not in stops and token not in punctuations]
#         #print("Number of word tokens:",len(word_tokens))
#     return (word_tokens)

# def text_tokenizing_to_sentences(text, remove_stopwords):
#     """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
#     tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#     raw_sentences = tokenizer.tokenize(text.strip())
#     sentences = []
#     for raw_sentence in raw_sentences:
#         if len(raw_sentence) > 0:
#             sentences.append(text_tokenizing(raw_sentence, remove_stopwords))
#             #print("Number of sentences:",len(sentences))
#     return sentences


# def createFeatureVector(words, model, num_features):
   
#     featureVector = np.zeros((num_features,),dtype="float32")

#     num_words = 0.

#     index2word_set = set(model.wv.index_to_key)
#     for word in words:
#         if word in index2word_set:
#             num_words += 1
#             featureVector = np.add(featureVector,model[word])  

#     if num_words != 0.:
#       featureVector = np.divide(featureVector,num_words)

#     return featureVector

# def getAverageFeatureVectors(essays, model, num_features):
   
#     i = 0
#     featureVec = np.zeros((len(essays),num_features),dtype="float32")

#     for essay in essays:
#         featureVec[i] = createFeatureVector(essay, model, num_features)
#         i = i + 1
#     return featureVec

  
# cv = KFold( n_splits=5, shuffle=True)
# results = []
# y_pred_list = []
# count = 1

# import pandas as pd

# df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
# score = df['rater1_domain1']
# for traincv, testcv in cv.split(df):
#     print("\n--------Fold {}--------\n".format(count))
#     X_test, X_train, y_test, y_train = df.iloc[testcv], df.iloc[traincv], score.iloc[testcv], score.iloc[traincv]
#     train_essays = X_train['essay']
#     test_essays = X_test['essay']
#     sentences = []
#     for essay in train_essays:
#         sentences += text_tokenizing_to_sentences(essay, remove_stopwords = True)
#     num_workers = 4
#     num_features = 300 
#     min_word_count = 40
#     context = 10
#     downsampling = 1e-3
#     model = Word2Vec(sentences, workers=num_workers, vector_size =num_features, min_count = min_word_count, window = context, sample = downsampling)
#     model.init_sims(replace=True)
#     model.wv.save_word2vec_format('word2vecmodel.bin', binary=True)
#     clean_train_essays = []
#     for text in train_essays:
#         clean_train_essays.append(text_tokenizing(text, remove_stopwords=True))
#     trainDataVecs = getAverageFeatureVectors(clean_train_essays, model, num_features)
#     clean_test_essays = []
#     for text in test_essays:
#         clean_test_essays.append(text_tokenizing( text, remove_stopwords=True ))
#     testDataVecs = getAverageFeatureVectors( clean_test_essays, model, num_features )
#     trainDataVecs = np.array(trainDataVecs)
#     testDataVecs = np.array(testDataVecs)
#     trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
#     testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
#     lstm_model = get_model()
#     lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=1)
#     y_pred = lstm_model.predict(testDataVecs)
#     # if count == 5:
#     #     lstm_model.save('./model_weights/final_lstm.h5')
#     lstm_model.save_weights('./checkpoints/my_checkpoint')
#     y_pred = np.around(y_pred)
#     count += 1
        
# def get_model():
#     model = Sequential()
#     model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
#     model.add(LSTM(64, recurrent_dropout=0.4))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='relu'))
#     model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
#     return model


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/essay")
def get_essay(essay: str):
    spelling = grammar_spelling_mistakes(essay)
    
    # clean_test_essays = []
    # clean_test_essays.append(text_tokenizing( essay, remove_stopwords=True ))
   
    # testDataVecs = getAverageFeatureVectors( clean_test_essays, model, 300 )
    # testDataVecs = np.array(testDataVecs)
    # testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    # lstm_model = get_model()
    # lstm_model.load_weights('./checkpoints/my_checkpoint')
    # preds = lstm_model.predict(testDataVecs)
    # essay =  np.round_(preds,decimals=5)
    essay = 5
    return {"spelling": spelling, "essay": essay}
