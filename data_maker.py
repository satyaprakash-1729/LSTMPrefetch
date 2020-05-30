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

def get_bin(value, start1, end1, start2, end2, start3, end3, binsize1, binsize2, binsize3, num_classes=3000):
	if value > end3 or value < start1:
		return num_classes
	if value <= end1 and value >= start1:
		return int((value-start1+binsize1-0.01)/binsize1)
	if value <= end2 and value >= start2:
		return int((value-start2+binsize2-0.01)/binsize2)+(num_classes//3)
	if value <= end3 and value >= start3:
		return int((value-start3+binsize3-0.01)/binsize3)+((2*num_classes)//3)


def create_dataset(data, maxlen=10, data_cnt=50000, num_classes=3000):

	scaler = MinMaxScaler((0, 10))

	deltas = data["delta"].astype(float).values[:data_cnt*10]
	ips = data["ip"].astype(float).values[:data_cnt*10]
	addrs = data["addr"].astype(float).values[:data_cnt*10]

	deltas = deltas.reshape(-1, 1)
	kmeans = KMeans(n_clusters=3, random_state=0).fit(deltas)
	deltas = deltas.reshape(-1)

	range1 = deltas[kmeans.labels_==0]
	start1, end1 = min(range1), max(range1)
	range2 = deltas[kmeans.labels_==1]
	start2, end2 = min(range2), max(range2)
	range3 = deltas[kmeans.labels_==2]
	start3, end3 = min(range3), max(range3)

	print(start1, end1)
	print(start2, end2)
	print(start3, end3)
	binsize1 = 3*(end1-start1)/num_classes
	binsize2 = 3*(end2-start2)/num_classes
	binsize3 = 3*(end3-start3)/num_classes

	print(binsize1, binsize2, binsize3)

	deltas = [get_bin(delt, start1, end1, start2, end2, start3, end3, binsize1, binsize2, binsize3, num_classes) for delt in deltas]

	ips = ips.reshape(-1, 1)
	ips = scaler.fit_transform(ips)
	ips = ips.reshape(-1)

	addrs = addrs.reshape(-1, 1)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(addrs)
	addrs = addrs.reshape(-1)

	cluster_ids = kmeans.labels_[:]

	firstclass = addrs[cluster_ids==0]
	secondclass = addrs[cluster_ids==1]
	# thirdclass = deltas[kmeans.labels_==2]

	firstclass = firstclass.reshape(-1, 1)
	secondclass = secondclass.reshape(-1, 1)
	# thirdclass = thirdclass.reshape(-1, 1)
	firstclass = scaler.fit_transform(firstclass)
	secondclass = scaler.fit_transform(secondclass)
	# thirdclass = scaler.fit_transform(thirdclass)
	firstclass = firstclass.reshape(-1)
	secondclass = secondclass.reshape(-1)
	# thirdclass = thirdclass.reshape(-1)

	addrs[cluster_ids==0] = firstclass
	addrs[cluster_ids==1] = secondclass
	# deltas[cluster_ids==2] = thirdclass

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
		X.append(list(zip(ng1[indx][:-1], ng4[indx][:-1], ng3[indx][:-1])))
		y.append(ng2[indx][-1])

	y = to_categorical(y, num_classes=num_classes+1)
	return np.array(X).reshape(-1, maxlen, 3), np.array(y)


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def get_lstm_model(X, y, rnn_units=32, batch_size=64, maxlen=10, num_labels=3, num_classes=3000):
	input_layer1 = tf.keras.layers.Input(shape=(maxlen, 3, ), name='input1')
	lstm_layer1 = tf.keras.layers.LSTM(rnn_units, activation='relu')(input_layer1)
	prediction = tf.keras.layers.Dense(num_classes+1, activation='softmax')(lstm_layer1)

	model = tf.keras.models.Model(inputs=input_layer1, outputs=prediction)
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard_callback])
	return model

# parse_data()
data = get_and_analyze_data(start_perc=0.05, plot=False)
X, y = create_dataset(data, data_cnt=100000, num_classes=10000)
model = get_lstm_model(X, y, rnn_units=64, batch_size=64, num_classes=10000)
# parse_data()