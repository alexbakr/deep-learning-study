import json
import os
import random
import sys
import requests

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

from utils import load_model

MARKET_CHECK_API_KEY = 'nhxAhnIxAUwXyc0qDINPgmTn2HYoIy0n'
MARKET_CHECK_API_URL = 'http://api.marketcheck.com/v1/stats?api_key=%s&ymm=%d|%s|%s'

def load_value_from_prediction(prediction):
	split_pred = prediction.split(',')
	split_data = split_pred[0].split(' ')
	make = split_data[0]
	model = split_data[1:-2]
	year = int(split_data[-1])
	
	formatted_url = MARKET_CHECK_API_URL % (MARKET_CHECK_API_KEY, year, make, "%20".join(model))
	print(f'Request URL: {formatted_url}')
	
	headers = {
		'host': 'marketcheck-prod.apigee.net'
	}
	
	response = requests.request('GET', formatted_url, headers=headers)
	mean_price = response.json()['price_stats']['geometric_mean']
	
	print(f'Mean Price: ${mean_price}')
	
	

if __name__ == '__main__':
	if sys.argv.index('--i') == -1 or len(sys.argv) < sys.argv.index('--i') + 1:
		print('No image supplied!')
		quit()
	
	print(sys.argv)
	
	image_path = sys.argv[sys.argv.index('--i') + 1]
	print(image_path)

	img_width, img_height = 224, 224
	model = load_model()
	model.load_weights('models/model.96-0.89.hdf5')

	cars_meta = scipy.io.loadmat('devkit/cars_meta')
	class_names = cars_meta['class_names']  # shape=(1, 196)
	class_names = np.transpose(class_names)
	
	print('Start processing image: {}'.format(image_path))
	bgr_img = cv.imread(image_path)
	bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
	rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
	rgb_img = np.expand_dims(rgb_img, 0)
	preds = model.predict(rgb_img)
	prob = np.max(preds)
	class_id = np.argmax(preds)
	prediction = class_names[class_id][0][0]
	text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
	print('\n\n')
	print(text)
	load_value_from_prediction(prediction)
	
