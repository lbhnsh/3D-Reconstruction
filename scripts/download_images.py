import requests
from PIL import Image
from io import BytesIO
import os

# URL of the image you want to download
# image_url = 'https://plus.unsplash.com/premium_photo-1676517307840-944e0e923edf?auto=format&fit=crop&q=80&w=2071&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
image_url='http://images.cocodataset.org/val2017/000000039769.jpg'
# Send a GET request to the URL to download the image
response = requests.get(image_url)

# Check if the request was successful
if response.status_code == 200:
    # Use PIL to open the image from the response content
    image = Image.open(BytesIO(response.content))
    
    # Optionally, you can save the image
    image.save('/home/labhansh/Open3D/MiDaS/input/cat_og.jpg')
    # Or perform some operations on the image
    # For example, you can display the image
    image.show()
else:
    print(f"Failed to download the image. Status code: {response.status_code}")

