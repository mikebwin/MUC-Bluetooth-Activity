from numpy import genfromtxt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy.optimize import minimize
import math
import traceback
import numpy
from numpy import genfromtxt
import csv
import random

rssi_data_buffer = []
data_buffer_size = 5
global knn_model_x 
global knn_model_y 
global knn_model_z 
global live_data


def initialize_knn_model(path_to_crowd_sourced_data, num, distance_metric="minkowski"):
	''' load the crowd sourced data file and train the knn regressor

    Arguments:
        path_to_crowd_sourced_data {String} -- Path to crowd source data
    '''
	global live_data
	global knn_model_x
	global knn_model_y
	global knn_model_z
	if len(path_to_crowd_sourced_data) == 0:
		print('Please provide path_to_crowd_sourced_data in indoor_localization_server.py')
		return
	try:
		# crowd_sourced_data = genfromtxt(path_to_crowd_sourced_data, delimiter=',',
		#                                 dtype="S23, f, f, f, f, f, f, f, f, f, f, f", autostrip=True)
		try:
			live_data, processed_data = preprocess_data_for_knn(path_to_crowd_sourced_data)
			knn_model_x, knn_model_y, knn_model_z = build_knn_model(processed_data, num, distance_metric)
		except Exception as e:
			print('[initialize_knn_model] An error occured when preparing knn model')
			print(e.__doc__)
			print(traceback.print_exc())

	except Exception as e:
		print('[initialize_knn_model] An error occured when loading crowd_sourced_data')
		print(e.__doc__)
		print(traceback.print_exc())


def receive_and_process_live_data(rssi_data):
	''' receive, process and then perform calcualtion on new rssi values.
    Get called when a new rssi_data is available form the node.js BLE
    scanning script.
    Arguments:
        rssi_data {list} -- A list with a size that is the same as the
        the number of beacon used for the activity. Index of the entry
        corespond to the beacon number id.
    TODO:
        1. maintain a buffer of incomming data of a fixed size
        2. process the buffer data to get a single value
    hint:
        For received rssi_data, think about how you want to handle it?
        If it has a few missing/invalid values, what is the cause of
        that?
        How do you want to handle it?
        If it has a lot of missing/invalid values, how do you want to
        handle it?
    '''

	print('\nreceive rssi data {0}'.format(rssi_data))
	''' The code below maintains rssi_data_buffer, a list that contains
        a list of most recent rssi reading.
    '''

	rssi_data_buffer.append(rssi_data)
	if len(rssi_data_buffer) <= data_buffer_size:
		pass
	else:
		rssi_data_buffer.pop(0)

	# TODO: Obtain a single value for each beacon from the rssi_data_buffer
	rssi_sum = [0, 0, 0, 0, 0, 0, 0, 0]
	for data in rssi_data_buffer:
		for index, num in enumerate(data):
			rssi_sum[index] += num

	processed_rssi = [float(x) / len(rssi_data_buffer) for x in rssi_sum]

	knn_location_x = -1
	knn_location_y = -1
	knn_location_z = -1
	try:
		knn_location_x, knn_location_y, knn_location_z = perform_knn_with_live_data(processed_rssi)
		# Send to node.js server, which will relay the message the visualization tool
		# s.send(json.dumps({'key':'knn_location', 'x':knn_location_x, 'y':knn_location_y, 'z':knn_location_z}) + '\n')
	except NotImplementedError:
		print('[receive_and_process_live_data] perform_knn_with_live_data not implemented')
	except Exception as e:
		print('[receive_and_process_live_data] An error occured when perform_knn_with_live_data')
		print(e.__doc__)
		print(traceback.print_exc())

	dists = [-1] * 8
	try:
		dists = rssi_to_dist(processed_rssi)
		print(dists)
		# Send to node.js server, which will relay the message the visualization tool
		# s.send(json.dumps({'key':'dists', 'dist_array': dists}) + '\n')
	except NotImplementedError:
		print('[receive_and_process_live_data] rssi_to_dist not implemented')
	except Exception as e:
		print('[receive_and_process_live_data] An error occured when rssi_to_dist')
		print(e.__doc__)
		print(traceback.print_exc())

	trilateration_location_x = -1
	trilateration_location_y = -1
	trilateration_location_z = -1
	try:
		trilateration_location_x, trilateration_location_y, trilateration_location_z = perform_trilateration_with_live_data(
			dists)
		# Send to node.js server, which will relay the message the visualization tool
		# s.send(json.dumps({'key':'trilateration_location', 'x':trilateration_location_x, 'y':trilateration_location_y, 'z':trilateration_location_z}) + '\n')
	except NotImplementedError:
		print('[receive_and_process_live_data] perform_trilateration_with_live_data not implemented')
	except Exception as e:
		print('[receive_and_process_live_data] An error occured when perform_trilateration_with_live_data')
		print(e.__doc__)
		print(traceback.print_exc())

	return [knn_location_x, knn_location_y, knn_location_z], dists, [trilateration_location_x, trilateration_location_y,
	                                                                 trilateration_location_z]


def preprocess_data_for_knn(path_to_crowd_sourced_data):
	'''Preprocess the crowd_sourced data
    How do you handle missing/invalid data(rssi = 0)? What is the
    casue of missing/invalid data?
    TODO: preprocess the crowd_sourced_data before using it to build the knn model
    Arguments:
        crowd_sourced_data {[type]} -- [description]
    '''
	# path_to_crowd_sourced_data = "Data/crowd_sourced_data.csv"
	with open(path_to_crowd_sourced_data, 'rt') as f:
		reader = csv.reader(f, skipinitialspace=True)
		crowd_sourced_data = list(reader)
	averages = {}
	# Find averages
	for entry in crowd_sourced_data:
		if entry[0] not in averages:
			averages[entry[0]] = [0, 0, 0, 0, 0, 0, 0, 0]
		for i in range(0, 8):
			averages[entry[0]][i] = averages[entry[0]][i] + float(entry[i + 1])
	for entry in averages:
		for i in range(0, 8):
			averages[entry][i] = averages[entry][i] / 10  # 10 entries per submission
	# Replace RSSI = 0 with -95 since 0 means that the beacon was too far away to pick up a signal
	for i in range(len(crowd_sourced_data)):
		for j in range(len(crowd_sourced_data[0])):
			if crowd_sourced_data[i][j] == '0':
				if (averages[crowd_sourced_data[i][0]][j - 1] == 0):
					crowd_sourced_data[i][j] = -95
	live_data = []
	for i in range(0,5):
		idxToRemove = random.randint(0, len(crowd_sourced_data))
		#print("Removing index: ", idxToRemove)
		live_data.append(crowd_sourced_data[idxToRemove])
		del crowd_sourced_data[idxToRemove]
	#Format return arrays
	for i in range(len(crowd_sourced_data)):
		RSSIs = [float(crowd_sourced_data[i][k]) for k in range(1, 9)]
		crowd_sourced_data[i] = [RSSIs, float(crowd_sourced_data[i][9]), float(crowd_sourced_data[i][10]),
		                         float(crowd_sourced_data[i][11])]
	for i in range(len(live_data)):
		RSSIs = [float(live_data[i][k]) for k in range(1, 9)]
		#print(live_data[i][9],live_data[i][10],live_data[i][11])
		live_data[i] = [RSSIs, float(live_data[i][9]), float(live_data[i][10]), float(live_data[i][11])]
	return (numpy.asarray(live_data), numpy.asarray(crowd_sourced_data))


def build_knn_model(processed_data, neighbors, dist_metric):
	# TODO: Refer to the scikit-learn documentation to see how to use KNeighborsRegressor
    knn_x = KNeighborsRegressor(n_neighbors=neighbors, metric=dist_metric)
    knn_y = KNeighborsRegressor(n_neighbors=neighbors, metric=dist_metric)
    knn_z = KNeighborsRegressor(n_neighbors=neighbors, metric=dist_metric)

    # TODO: How do you extract the information you need from processed data
    
    #Note: processed_data is a 2D array 
    rssi_data = []
    x_data = []
    y_data = []
    z_data = []
    for data in processed_data:
        rssi_data.append(data[0])
        x_data.append(data[1])
        y_data.append(data[2])
        z_data.append(data[3])

    knn_x.fit(rssi_data, x_data)
    knn_y.fit(rssi_data, y_data)
    knn_z.fit(rssi_data, z_data)

    # return knn_x, knn_y, knn_z
    return (knn_x, knn_y, knn_z)


def perform_knn_with_live_data(live_data):
	''' perform regression using the knn model you built:
        After you finish this method. The visualization should show
        the position calculated by fingerprinting using knn.
    Arguments:
        proccessed_live_rssi_data -- proccessed live rssi data you want to run predict with

    TODO: Finish this function. Look at scikit-learn's documentation
    to see how to use the knn model to predict
    '''

	global knn_model_x
	global knn_model_y
	global knn_model_z
	proccessed_live_rssi_data = []
	for data in live_data:
		proccessed_live_rssi_data.append(data[0])
	x = knn_model_x.predict(proccessed_live_rssi_data)
	y = knn_model_y.predict(proccessed_live_rssi_data)
	z = knn_model_z.predict(proccessed_live_rssi_data)
	#print(proccessed_live_rssi_data)
	#print(x)
	#print(y)
	#print(z)
	return x, y, z


def rssi_to_dist(proccessed_live_rssi_data):
	'''convert proccessed_live_rssi_data to distances to each beacon
    Free-space path loss model only works in ideal case. The reality
    is unfortunately different. In this acticity, we use something
    that is fitted with real data.
    Implement this: https://gist.github.com/eklimcz/446b56c0cb9cfe61d575
    After you finish implementin this method, the visualization should
    show circles around the the beacon to reflect the distance value
    you calculated
    Arguments:
        proccessed_live_rssi_data -- processed rssi. An Array of rssi
        representing the rssi to each beacon
    returns:
        A list representing the distance in meters of each beacon
    '''
	# TODO: implement the algorithm from the link above
	txPower = -54  # According to the spec of the beacon

	distances = []
	dist_to_beacons = [None]
	#print(len(proccessed_live_rssi_data))

	for rssi in proccessed_live_rssi_data:  # loop through if the processed data is 2d, i need to change if only 1d'
		dist_to_beacons = []
		for beacon in rssi[0]:
			#print(beacon)
			ratio = (beacon * 1.0) / txPower
			if (beacon == 0):
				dist_to_beacons.append(0)
			elif ((ratio) < 1.0):
				dist_to_beacons.append(ratio ** 10)
			else:
				distance = (0.89976) * (ratio ** 7.7095) + 0.111
				dist_to_beacons.append(distance)
		distances.append(dist_to_beacons)
	return distances


def perform_trilateration_with_live_data(distances):
	'''Perform trilateration calculation
    Given your distance to each of the beacon and the location of
    each beacon(below), calculate your location.
    After you implement this method. The visualization should show
    the position calculated by trilateration.
    Arguments:
        distances -- An array with your distance to each beacon
    Returns:
        x, y, z location of yourself
    '''
	coordinates = []
	for distance in distances:
		# x_i, y_i, z_i are the x, y, z coordinates of the beacons
		x_i = [13.369, 13.21, 10.066, 3.012, 3.884, 5.152, 6.024, 8.904]
		y_i = [13.396, 5.18, 3.911, 3.066, 10.067, 13.211, 8.958, 6.025]
		z_i = [4.824, 4.12, 3.723, 2.466, 3.723, 4.12, 1.871, 1.871]
		d_i = distance
		initial_guess = [8.719, 9.037, 1.534]

		# TODO: fill in the objective_function for optimization. Use SSE,
		# sum of squared error as the score you want to minimize

		def objective_function(xyz_guess, xyzd):
			sum = 0
			for i in range(len(xyzd[0])):
				sum_of_squares = pow(xyz_guess[0] - xyzd[0][i], 2) + pow(xyz_guess[1] - xyzd[1][i], 2) + pow(xyz_guess[2] - xyzd[2][i], 2)
				sum += pow(pow(sum_of_squares, .5) - xyzd[3][i], 2)
			# Multiply by w_i, whatever that is - lower weight for rssi of -90 bc can't trust it 
			return sum

		(x,y,z) = (None, None, None)
		try:
			x, y, z = minimize(objective_function, initial_guess, [x_i, y_i, z_i, d_i]).x
		except NotImplementedError:
			print('[perform_trilateration_with_live_data] objective function for minimization not implemented')
		except Exception as e:
			print('[perform_trilateration_with_live_data] An error occured when minimize')
			print(e.__doc__)
			print(traceback.print_exc())
		coordinates.append((x,y,z))
	return coordinates

if __name__ == '__main__':
	metrics = ["minkowski", "euclidean", "manhattan", "chebyshev"]

	# for i in metrics:
	# 	print i
	# 	initialize_knn_model('Data/crowd_sourced_data.csv', 5, i)
	# 	for item in perform_knn_with_live_data(live_data):
	# 		print item
	#
	# for i in range(1, 10):
	# 	print i
	# 	initialize_knn_model('Data/crowd_sourced_data.csv', i)
	# 	for item in perform_knn_with_live_data(live_data):
	# 		print item

	initialize_knn_model('Data/crowd_sourced_data.csv', 5)
	perform_knn_with_live_data(live_data)
	distances = rssi_to_dist(live_data)
	coordinates = perform_trilateration_with_live_data(distances)
	for coordinate in coordinates:
		print(coordinate)
