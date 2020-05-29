import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import ngrams
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow.keras.backend as K


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

def get_memory_addr_line(line):
	addr = 0
	rw = ""
	ip = 0
	tokens = line.split()
	if len(tokens)==5 or len(tokens)==6 and (tokens[1]=='R' or tokens[1]=='W'):
		try:
			addr = int(tokens[2], 16)
			rw = tokens[1]
			ip = int(tokens[0].strip(":"), 16)
		except:
			print("Error parsing...")
			return 0, "", 0
	return addr, rw, ip


def parse_data(FINAL_DATA_FILE = "data_processed.csv", final_data = {"delta": [], "rw": [], "ip": [], "addr": []}):
	with open("pinatrace.out", "r") as data_file:
		last_addr = 0
		lines = data_file.readlines()
		for line in lines:
			addr, rw, ip = get_memory_addr_line(line)
			if addr!=0:
				final_data["delta"].append(addr - last_addr)
				final_data["rw"].append(rw)
				final_data["ip"].append(ip)
				final_data["addr"].append(addr)
				last_addr = addr

	df = pd.DataFrame(final_data)
	df.to_csv(FINAL_DATA_FILE, index=False, sep='\t')

from sklearn.cluster import KMeans

def get_and_analyze_data(DATA_FILE="data_processed.csv", start_perc=0, max_cnt=50000, plot=False):
	data = pd.read_csv(DATA_FILE, sep='\t')
	start_pos = int(data.shape[0] * (start_perc))
	data1 = data[start_pos:]
	print("Unique PCs: ", data["ip"].nunique())
	if plot:
		start_pos = int(data.shape[0] * (start_perc))
		y = data["delta"].astype(float).values[start_pos:start_pos+max_cnt]
		yaddr = data["addr"].astype(float).values[start_pos:start_pos+max_cnt]
		x = np.array([i for i in range(len(y))])

		y = y.reshape(-1, 1)
		kmeans = KMeans(n_clusters=3, random_state=0).fit(y)
		y = y.reshape(-1)

		first = y[kmeans.labels_==0]
		second = y[kmeans.labels_==1]
		third = y[kmeans.labels_==2]

		firstx = x[kmeans.labels_==0]
		secondx = x[kmeans.labels_==1]
		thirdx = x[kmeans.labels_==2]

		plt.figure(1)
		plt.title("Delta over time")
		plt.xlabel("reference index")
		plt.ylabel("Delta")
		# plt.plot(x, y, 'r.')
		plt.plot(firstx, first, 'r.')
		plt.plot(secondx, second, 'g.')
		plt.plot(thirdx, third, 'b.')
		plt.show()

		plt.figure(2)
		plt.title("Addr over time")
		plt.xlabel("Index")
		plt.ylabel("Addr")
		plt.show()

	return data1


def create_dataset(data, maxlen=10, data_cnt=5000):

	scaler = MinMaxScaler((0, 10))

	deltas = data["delta"].astype(float).values[:data_cnt*10]
	ips = data["ip"].astype(float).values[:data_cnt*10]

	ips = ips.reshape(-1, 1)
	ips = scaler.fit_transform(ips)
	ips = ips.reshape(-1)

	deltas = deltas.reshape(-1, 1)
	kmeans = KMeans(n_clusters=3, random_state=0).fit(deltas)
	deltas = deltas.reshape(-1)

	cluster_ids = kmeans.labels_[:]

	firstclass = deltas[kmeans.labels_==0]
	secondclass = deltas[kmeans.labels_==1]
	thirdclass = deltas[kmeans.labels_==2]

	firstclass = firstclass.reshape(-1, 1)
	secondclass = secondclass.reshape(-1, 1)
	thirdclass = thirdclass.reshape(-1, 1)
	firstclass = scaler.fit_transform(firstclass)
	secondclass = scaler.fit_transform(secondclass)
	thirdclass = scaler.fit_transform(thirdclass)
	firstclass = firstclass.reshape(-1)
	secondclass = secondclass.reshape(-1)
	thirdclass = thirdclass.reshape(-1)

	deltas[cluster_ids==0] = firstclass
	deltas[cluster_ids==1] = secondclass
	deltas[cluster_ids==2] = thirdclass

	ng1 = ngrams(ips, maxlen+1)
	ng2 = ngrams(deltas, maxlen+1)
	ng3 = ngrams(cluster_ids, maxlen+1)

	ng1 = [ngg for ngg in ng1]
	ng2 = [ngg for ngg in ng2]
	ng3 = [ngg for ngg in ng3]
	inds = np.random.choice([i for i in range(len(ng1))], data_cnt, replace=False)
	
	ng1 = np.array(ng1)[inds]
	ng2 = np.array(ng2)[inds]
	ng3 = np.array(ng3)[inds]

	X = []
	y = []
	for indx, _ in enumerate(ng1):
		X.append(list(zip(ng1[indx][:-1], ng2[indx][:-1], ng3[indx][:-1])))
		y.append(ng2[indx][-1])

	return np.array(X).reshape(-1, maxlen, 3), np.array(y)


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def get_lstm_model(X, y, rnn_units=32, batch_size=64, maxlen=10, num_labels=3):
	input_layer1 = tf.keras.layers.Input(shape=(maxlen, 3, ), name='input1')
	lstm_layer1 = tf.keras.layers.LSTM(rnn_units, activation='relu')(input_layer1)
	prediction = tf.keras.layers.Dense(1)(lstm_layer1)

	model = tf.keras.models.Model(inputs=input_layer1, outputs=prediction)
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='mse', metrics=[coeff_determination])
	print(model.summary())

	model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard_callback])
	return model

# parse_data()
data = get_and_analyze_data(start_perc=0.005, plot=True)
# X, y = create_dataset(data, data_cnt=100000)
# model = get_lstm_model(X, y, rnn_units=64, batch_size=256)
# parse_data()