# importing the requests library
import argparse
import requests

# defining the api-endpoint
API_ENDPOINT = "http://localhost:5000/house_number_predict/predict/"

# taking input image via command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
image_data = open(image_path, 'rb').read()
data = {'image_data': image_data}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, files=data)

# extracting the response
print("{}".format(r.text))


