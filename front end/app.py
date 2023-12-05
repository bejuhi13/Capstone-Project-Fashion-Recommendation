from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import time
import model
from scrape import Scraper
import os

app = Flask(__name__)
mdl = model.Model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint for detecting objects in an image.

    Returns:
        JSON: A JSON response containing the detected objects and their details.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Process the image using your pre-trained model
    #create directory if not exists
    if not os.path.exists('./dlimages'):
        os.mkdir('dlimages')
    file.save('dlimages/'+file.filename)
    
    item_lst, bboxlst = mdl.run_model('dlimages/'+file.filename)
    os.remove('dlimages/'+file.filename)
    
    res_lst = []
    for i in range(len(item_lst)):
        show_tag = {}
        webp = 'myntra' #CHANGE MODEL RESULTS (HNM CANT DETECT IF THE ITEM DESCRIPTION STRING IS APPENDED IN THE FRONT)
        result = Scraper().get_results(item_lst[i],1, webpage=webp)[webp][0]  #ONLY ONE RESULT IS SUPPORTED ON THE FRONT-END SO KEEP THE [0] AT THE END
        show_tag['box'] = bboxlst[i]
        show_tag['Item'] = item_lst[i].title()
        show_tag['Name'] = result['name']
        show_tag['ProductId'] = result['productid']
        show_tag['Price'] = result['price']
        show_tag['link'] = result['link']
        try:
            show_tag['Colour'] = result['colour']
        except:
            pass
        res_lst.append(show_tag)
        
    #res_lst = [{'name': 'Printed rugby shirt', 'productId': '1209720003', 'price': '1499.0', 'colour': 'Cream/Los Angeles', 'link': 'https://www2.hm.com/en_in/productpage.1209720003.html', 'box': [47.165672302246094, 82.7443618774414, 263.94921875, 335.1565856933594], 'Item': 'White Short Sleeve Top'}, {'name': 'Crinkled loungewear trousers', 'productId': '1185975002', 'price': '2299.0', 'colour': 'Dark green', 'link': 'https://www2.hm.com/en_in/productpage.1185975002.html', 'box': [77.82475280761719, 308.02825927734375, 243.9527130126953, 678.34423828125], 'Item': 'Teal Trousers'}]
 
    print(res_lst)
    
    response = {
        'objects': res_lst
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)
