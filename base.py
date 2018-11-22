import pandas as pd
import jieba
import re
import numpy as np
from keras.utils import *
from keras.preprocessing import sequence
from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

def tokenize(data):
	return [x.strip().lower() for x in re.split('(\W+)', data) if x.strip()]

labels = {'unrelated':0,'agreed':1,'disagreed':2}
labels_reverse = {0:'unrelated',1:'agreed',2:'disagreed'}
new_data = pd.DataFrame()
new_data['gold_label'] = data['label']
new_data['sentence1'] = data['title1_zh']
new_data['sentence2'] = data['title2_zh'].astype(str)
#print(new_data['sentence1'])
new_data['sentence1'] = new_data['sentence1'].apply(lambda x:' '.join(jieba.cut(x)))
new_data['sentence2'] = new_data['sentence2'].apply(lambda x:' '.join(jieba.cut(x)))
#print(new_data['sentence3'])

tests = pd.DataFrame()
tests['id'] = data['id']
tests['sentence1'] = test_data['title1_zh'].astype(str)
tests['sentence2'] = test_data['title2_zh'].astype(str)
tests['sentence1'] = test_data['title1_zh'].apply(lambda x:' '.join(jieba.cut(x)))
tests['sentence2'] = test_data['title2_zh'].apply(lambda x:' '.join(jieba.cut(x)))

word_dict = {}
word_dict['PAD'] = len(word_dict)
word_dict['UNK'] = len(word_dict)

for q1,q2 in zip(new_data['sentence1'], new_data['sentence2']):
	q1 = tokenize(q1)
	q2 = tokenize(q2)
	for word in q1:
		if word not in word_dict:
			word_dict[word] = len(word_dict)
	for word in q2:
		if word not in word_dict:
			word_dict[word] = len(word_dict)	
			
def get_vectors(word_dict, vec_file, emb_size=300):
	word_vectors = np.random.uniform(-0.1, 0.1, (len(word_dict), emb_size))
	
	f = open(vec_file, 'r', encoding='utf-8')
	vec = {}
	for line in f:
		line = line.split()
		vec[line[0]] = np.array([float(x) for x in line[-emb_size:]])
	f.close()  
	for key in word_dict:
		low = key.lower()
		if low in vec:
			word_vectors[word_dict[key]] = vec[low]
				
	return word_vectors	
	
		
embeddings_chinese = "../aaai2019/data/wiki.zh.vec"			
word_embeddings = get_vectors(word_dict, embeddings_chinese)
print(len(word_embeddings))

max_len_sentence = 30
def map_to_id(data, vocab):
	return [vocab[word] if word in vocab else 1 for word in data]
	
def load_data(data, word_dict, labels=labels):
	X,Y,Z = [], [], []
	for label, q1, q2 in zip(data['gold_label'],data['sentence1'],data['sentence2']):
		q1 = map_to_id(tokenize(q1), word_dict)
		q2 = map_to_id(tokenize(q2), word_dict)	
		if len(q1) > max_len_sentence:
			q1 = q1[:max_len_sentence]
		if len(q2) > max_len_sentence:
			q2 = q2[:max_len_sentence]
		X+= [q1]
		Y+= [q2]
		Z+= [labels[label]]		

	X = sequence.pad_sequences(X, maxlen = max_len_sentence)
	Y = sequence.pad_sequences(Y, maxlen = max_len_sentence)
	Z = to_categorical(Z,num_classes=3)

	return X, Y, Z
	
def load_test_data(data, word_dict):
	X,Y = [], []
	for q1, q2 in zip(data['sentence1'],data['sentence2']):
		q1 = map_to_id(tokenize(q1), word_dict)
		q2 = map_to_id(tokenize(q2), word_dict)	
		if len(q1) > max_len_sentence:
			q1 = q1[:max_len_sentence]
		if len(q2) > max_len_sentence:
			q2 = q2[:max_len_sentence]
		X+= [q1]
		Y+= [q2]
	X = sequence.pad_sequences(X, maxlen = max_len_sentence)
	Y = sequence.pad_sequences(Y, maxlen = max_len_sentence)
	return X, Y
	
def submult(input_1, input_2):
    mult = Multiply()([input_1, input_2])
    sub = Lambda(lambda x: K.abs(x[0] - x[1]))([input_1,input_2])
    out_= Concatenate()([sub, mult])
    return out_

new_data, val_data = train_test_split(new_data, test_size = 0.2, stratify=new_data['gold_label'], random_state=0)	
X, Y, Z = load_data(new_data, word_dict)
devX, devY, devZ = load_data(val_data, word_dict)
testX, testY = load_test_data(tests, word_dict)


lstm_dimension = 100

source_embeddings = Embedding(len(word_dict), 300, weights=[word_embeddings], input_length=(max_len_sentence,), trainable=False)

source_input1 = Input(shape=(max_len_sentence,))
source_input2 = Input(shape=(max_len_sentence,))


source_q1 = source_embeddings(source_input1)
source_q2 = source_embeddings(source_input2)


source_lstm = Bidirectional(CuDNNLSTM(lstm_dimension, return_sequences=True))

source_q1_encoded = GlobalMaxPooling1D()(source_lstm(source_q1)) 
source_q2_encoded = GlobalMaxPooling1D()(source_lstm(source_q2))


source_merged = Concatenate()([source_q1_encoded, source_q2_encoded, submult(source_q1_encoded, source_q2_encoded)])

def classifier(merged):
	dense = Dense(1000, activation='tanh')(merged)
	#dense = Dropout(dropout_rate)(dense)
	out = Dense(3, activation='softmax', name="cnli")(dense)
	return out
	
source_out = classifier(source_merged)

model = Model([source_input1,source_input2], [source_out])
print(model.summary())
model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])

results = model.fit([X, Y], [Z], 
				validation_data = ([devX, devY], [devZ]),
				batch_size = 1024,
				shuffle = True,
				epochs=15)

preds = model.predict([testX,testY])
preds = np.argmax(preds, axis=1)

submits = pd.DataFrame()
submits['Id'] = test['id']
submits['Category'] = preds
submits['Category'] = submits['Category'].apply(lambda x: labels_reverse[x])
submits.to_csv('result.txt',index=False)

