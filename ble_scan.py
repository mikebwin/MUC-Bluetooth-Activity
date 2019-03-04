from beacontools import BeaconScanner
import requests
import time

def callback(bt_addr, rssi, packet, additional_info):
    if additional_info['uuid'] == 'fda50693-a4e2-4fb1-afcf-c6eb07647825':
        r = requests.post('http://127.0.0.1:8090/ble_scan', json = {'name': 'Beacon ' + str(additional_info['major']), 'inquiryRssi':rssi})

if __name__ == '__main__':
    while 1:
        scanner = BeaconScanner(callback)
        scanner.start()
        time.sleep(3)
        scanner.stop()
