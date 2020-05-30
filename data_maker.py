import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import ngrams
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow.keras.backend as K
from keras.utils import to_categorical
from collections import Counter


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

def get_and_analyze_data(DATA_FILE="data_processed.csv", start_perc=0.1, max_cnt=500, plot=False):
	data = pd.read_csv(DATA_FILE, sep='\t')
	start_pos = int(data.shape[0] * (start_perc))
	data1 = data[start_pos:]
	print("Unique PCs: ", data["ip"].nunique())
	print("Unique Deltas: ", data["delta"].nunique())


	# n, bins, patches = plt.hist(data["delta"].astype(float).values[start_pos:start_pos+max_cnt], density=True, bins=1000)
	# print("n --> ", n)
	# print("bins --> ", bins)
	# print("patches --> ", patches)
	# print()
	# for patch in patches:
	# 	print(patch.get_extents())
	# 	print(patch.get_width())
	# 	print(patch.get_window_extent())
	# 	break

	if plot:
		start_pos = int(data.shape[0] * (start_perc))
		y = data["delta"].astype(float).values[start_pos:start_pos+max_cnt]
		yaddr = data["addr"].astype(float).values[start_pos:start_pos+max_cnt]
		x = np.array([i for i in range(len(y))])

		yaddr = yaddr.reshape(-1, 1)
		kmeans = KMeans(n_clusters=2, random_state=0).fit(yaddr)
		yaddr = yaddr.reshape(-1)

		first = yaddr[kmeans.labels_==0]
		second = yaddr[kmeans.labels_==1]

		firstx = x[kmeans.labels_==0]
		secondx = x[kmeans.labels_==1]

		plt.figure(1)
		plt.title("Address over time")
		plt.xlabel("Index")
		plt.ylabel("Address")
		plt.plot(firstx, first, 'r.')
		plt.plot(secondx, second, 'b.')
		plt.show()

		plt.figure(2)
		plt.title("Delta over time")
		plt.xlabel("Index")
		plt.ylabel("Delta")
		plt.plot(x, y, 'k.')
		plt.show()

		# plt.figure(3)
		# plt.title("Hist for cluster id = 1")
		# plt.hist(y[kmeans.labels_==1], bins=300)
		# plt.show()

	return data1


def create_dataset(data, maxlen=10, data_cnt=50000, num_classes=3000):

	scaler = MinMaxScaler((0, 10))

	deltas = data["delta"].astype(float).values[:data_cnt*100]
	ips = data["ip"].astype(float).values[:data_cnt*100]
	addrs = data["addr"].astype(float).values[:data_cnt*100]

	del_freq = Counter(deltas)
	max10000 = del_freq.most_common(num_classes)
	del_list = {}
	for idx, delta in enumerate(max10000):
		del_list[delta[0]] = idx

	ips = ips.reshape(-1, 1)
	ips = scaler.fit_transform(ips)
	ips = ips.reshape(-1)

	addrs = addrs.reshape(-1, 1)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(addrs)
	addrs = addrs.reshape(-1)

	cluster_ids = kmeans.labels_[:]

	firstclass = addrs[cluster_ids==0]
	secondclass = addrs[cluster_ids==1]

	firstclass = firstclass.reshape(-1, 1)
	secondclass = secondclass.reshape(-1, 1)

	firstclass = scaler.fit_transform(firstclass)
	secondclass = scaler.fit_transform(secondclass)

	firstclass = firstclass.reshape(-1)
	secondclass = secondclass.reshape(-1)

	addrs[cluster_ids==0] = firstclass
	addrs[cluster_ids==1] = secondclass

	ng1 = ngrams(ips, maxlen+1)
	ng2 = ngrams(deltas, maxlen+1)
	ng3 = ngrams(cluster_ids, maxlen+1)
	ng4 = ngrams(addrs, maxlen+1)

	ng1 = [ngg for ngg in ng1]
	ng2 = [ngg for ngg in ng2]
	ng3 = [ngg for ngg in ng3]
	ng4 = [ngg for ngg in ng4]
	inds = np.random.choice([i for i in range(len(ng1))], data_cnt, replace=False)
	
	ng1 = np.array(ng1)[inds]
	ng2 = np.array(ng2)[inds]
	ng3 = np.array(ng3)[inds]
	ng4 = np.array(ng4)[inds]

	X = []
	y = []
	for indx, _ in enumerate(ng1):
		if ng2[indx][-1] in del_list:
			X.append(list(zip(ng1[indx][:-1], ng4[indx][:-1], ng3[indx][:-1])))
			y.append(del_list[ng2[indx][-1]])

	y = to_categorical(y, num_classes=num_classes)
	return np.array(X).reshape(-1, maxlen, 3), np.array(y)


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def get_lstm_model(X, y, rnn_units=32, batch_size=64, maxlen=10, num_labels=3, num_classes=3000):
	input_layer1 = tf.keras.layers.Input(shape=(maxlen, 3, ), name='input1')
	lstm_layer1 = tf.keras.layers.LSTM(rnn_units, activation='relu')(input_layer1)
	prediction = tf.keras.layers.Dense(num_classes, activation='softmax')(lstm_layer1)

	model = tf.keras.models.Model(inputs=input_layer1, outputs=prediction)
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	model.fit(X, y, batch_size=64, epochs=3, validation_split=0.1, callbacks=[tensorboard_callback])
	return model

# parse_data()
data = get_and_analyze_data(start_perc=0.05, plot=False)
X, y = create_dataset(data, data_cnt=200000, num_classes=2000, maxlen=20)
model = get_lstm_model(X, y, rnn_units=128, batch_size=256, num_classes=2000, maxlen=20)
# parse_data()