import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
from tensorflow.keras.models import load_model
from colorthief import ColorThief
from tensorflow.keras.preprocessing import image
import numpy as np


def send_to_yolo(img):
    # Send image to YOLOv5 for object detection.

    # Path to the saved custom YOLOv5 model
    custom_model_path = 'C:/Users/bejuh/Juhi/ACADEMIC/Capstone/demo/best.pt'

    # Load the custom YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=custom_model_path)

    img_path = img

    # Detect objects in the image
    results = model(img_path)

    saved_class = []
    saved_bbox = []

    for pred in results.xyxy[0]:
        conf, cls, bbox = pred[4], pred[5], pred[:4]
        print(conf.item(),cls.item(),bbox.tolist())
        saved_class.append(cls.item())
        saved_bbox.append(bbox.tolist())

    # Show the original image with bounding boxes
    #img = results.show()
    print(results)
    return saved_class,saved_bbox

#when calling call in loop for bbox in saved_bbox
def send_to_crop(bbox, img):
    #bbox = [27.19845199584961, 128.70455932617188, 300.4272766113281, 444.82781982421875]

    # Load the image
    image_path = img
    original_image = Image.open(image_path)

    # Crop the image using the bounding box
    cropped_image = original_image.crop(bbox)

    # Save or display the cropped image
    #cropped_image.save('cropped_image4.png')
    #cropped_image.show()
    print(cropped_image)
    # Display the image inline (for Jupyter Notebooks)
    #plt.imshow(cropped_image)
    #plt.axis('off')
    return cropped_image

def send_to_cnn(cropped_image):
    modelcnn = load_model("C:/Users/bejuh/Juhi/ACADEMIC/Capstone/demo/inceptionv3_image_classifier_model.h5")
    
    img_path = cropped_image
    """img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 """ # Normalize the pixel values to be between 0 and 1

    # Make predictions
    predictions = modelcnn.predict(img_path)

    # Interpret the predictions (this depends on your specific model and task)
    # For example, if it's a classification model, you might want to get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)

    print("Predicted class:", predicted_class)
    # Assuming train_generator is your ImageDataGenerator
    class_indices = ['OTHER', 'animal','cartoon','chevron','floral','geometry','houndstooth','ikat','letter_numb','plain','polka dot','scales', 'skull','squares','stars','stripes','tribal']


    # Map the predicted class index to the class label
    predicted_class_label = class_indices[int(predicted_class)]

    # Create a ColorThief object
    color_thief = ColorThief(img_path)

    # Get the dominant color
    dominant_color = color_thief.get_color(quality=1)

    #print(f"Dominant Color: {dominant_color}")
    colours = ( (255, 255, 255, "white"),
            (255, 0, 0, "red"),
            (64, 0, 0, "dark red"),
            (0, 255, 0, "green"),
            (0,64,0, "dark green"),
            (0, 0, 255, "blue"),
            (0,0,64, "dark blue"),
            (255, 255, 0, "yellow"),
            (64,64,0, "olive green"),
            (0, 255, 255, "cyan"),
            (0,64,64, "teal"),
            (255, 0, 255, "magenta"),
            (128,0,128, "purple"),
            (0, 0, 0, "black"),
            (128,128,128,"grey"),
            (252, 102, 0, "orange"),
            (255, 192, 203, "light pink"),
            (173, 20, 87,"dark pink"),
            (59, 39, 12, "brown"),
            (229, 211, 179, "beige"),
            (128, 0, 0, "maroon"),
            )


    def nearest_colour( subjects, query ):
        return min( subjects, key = lambda subject: sum( (s - q) ** 2 for s, q in zip( subject, query ) ) )

    nearest_col  = str(nearest_colour( colours, dominant_color )[-1] )

    print(f"Nearest Color: {nearest_col}")


    #print("Predicted class label:", predicted_class_label)
    return predicted_class_label, nearest_col

#when calling call per item in main image
def send_to_webscrape(nearest_col, predicted_class_label, saved_class):
    class_label = int(saved_class)

    # Example mapping of class indices to class names
    class_names =  ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear','vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress','vest dress', 'sling dress']

    # Print the corresponding class name
    if 0 <= class_label < len(class_names):
        class_name = class_names[class_label]
        #print(f"Class Label: {class_label}, Class Name: {class_name}")
    else:
        print(f"Invalid class label: {class_label}")

    search_term = nearest_col+" "+predicted_class_label+" "+class_name
    print(search_term)
    #return search_term


if __name__=="__main__":
    img = "demo/redstripes.png"
    saved_class, saved_bbox = send_to_yolo(img)
    for bbox,sclass in saved_bbox, saved_class:
        cropped_img = send_to_crop(bbox, img)
        predicted_class_label, nearest_col = send_to_cnn(cropped_img)
        send_to_webscrape(nearest_col, predicted_class_label, sclass)


