import gevent.monkey

gevent.monkey.patch_all()
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

create_urllib3_context()
from flask import Flask, request
from flask_socketio import SocketIO, emit
import platform
from location_calculation import receive_and_process_live_data, initialize_knn_model
from multiprocessing import Process
import subprocess
from subprocess import Popen, PIPE

# static_folder = '../indoor-localisation-visualization/dist/indoor-localisation-visualization'
# static_folder = './static'
app = Flask(__name__, static_url_path='')
socketio = SocketIO(app)
port_num = 8090

# TODO: Change path_to_crowd_sourced_data to your actual file location
path_to_crowd_sourced_data = 'Data/crowd_sourced_data.csv'


@app.route('/')
def index():
	return app.send_static_file('index.html')


@app.route('/ble_scan', methods=['POST'])
def new_ble_scan():
	if request.is_json:
		content = request.get_json()
		socketio.emit('bleScan', content, namespace='/sock', broadcast=True)
		socketio.sleep(0)
	return 'OK'


@socketio.on('connect', namespace='/sock')
def on_connect():
	print('on connect')


@socketio.on('disconnect', namespace='/sock')
def disconnect():
	print('on disconnect')


@socketio.on('rssi_update', namespace='/sock')
def on_rssi_update(rssi_data):
	knn_xyz, dists, tri_xyz = receive_and_process_live_data(rssi_data)
	print('fingerprintting: {0} distances: {1} trilateration: {2}'.format(knn_xyz, dists, tri_xyz))
	emit('location', {'knn': {'x': knn_xyz[0], 'y': knn_xyz[1], 'z': knn_xyz[2]}, 'dists': dists,
	                  'trilateration': {'x': tri_xyz[0], 'y': tri_xyz[1], 'z': tri_xyz[2]}})
	socketio.sleep(0)


def ble_scan():
	a = subprocess.Popen(['python ble_scan.py'], shell=True, stdout=PIPE, stderr=PIPE)


if __name__ == '__main__':
	for i in range(1, 10):
		print('server starting at port: ' + str(port_num + i))
		initialize_knn_model(path_to_crowd_sourced_data, i)

		if platform.system() == 'Linux':
			print('==============is linux!!! ========')
			ble_scan_p = Process(target=ble_scan)
			ble_scan_p.start()

		socketio.run(app, host='localhost', port=port_num, debug=False)
