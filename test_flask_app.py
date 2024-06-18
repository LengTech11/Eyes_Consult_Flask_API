import requests

# specify the URL
url = 'http://127.0.0.1:5000/predict'

# path to the image file
image_path = 'Test/cataract/_7_2330751.jpg'

# open the image file and prepare it for send it as a POST request
with open(image_path, 'rb') as f:
    files = {'file': (image_path, f, 'image/jpeg')}

    # send the POST request
    response = requests.post(url, files=files)

# print the response
print(response.json())