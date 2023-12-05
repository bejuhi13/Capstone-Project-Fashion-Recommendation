import torch
from pathlib import Path
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from colorthief import ColorThief
from tensorflow.keras.preprocessing import image
import numpy as np
import os

class Model():
    """
    A class representing a model for object detection and classification.

    Attributes:
        yolo_custom_model_path (str): The path to the custom YOLO model.
        yolo_model: The YOLO model for object detection.
        yolo_class_names: The class names for the YOLO model.
        modelcnn: The CNN model for image classification.
    """

    def __init__(self):
        """
        Initializes the Model class by loading the YOLO and CNN models.
        """
        self.yolo_custom_model_path = 'models/best.pt'
        self.yolo_model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=self.yolo_custom_model_path)
        self.yolo_class_names = self.yolo_model.names
        self.modelcnn = load_model("models/inceptionv3_image_classifier_model.h5")
        
    def run_model(self, img):
        """
        Runs the model on the given image.

        Args:
            img (str): The path to the input image.

        Returns:
            search_term_lst (list): A list of search terms generated by the model.
            bboxlst (list): A list of bounding boxes generated by the YOLO model.
        """
        classlst, bboxlst = self.send_to_yolo(img)
        cropped_image = self.send_to_crop(bboxlst, img)
        search_term_lst = []
        img_name = img.split('/')[-1].split('.')[0]
        for i in range(len(bboxlst)):
            predicted_class_label, nearest_col = self.send_to_cnn('features/'+img_name+str(i)+'.png')
            #search_term_lst.append(nearest_col+" "+classlst[i]) #HnM hack
            search_term_lst.append(predicted_class_label+" "+nearest_col+" "+classlst[i])
        self.clear_features()
        return search_term_lst, bboxlst
        
        
    def send_to_yolo(self, img_path):  
        """
        Sends the image to the YOLO model for object detection.

        Args:
            img_path (str): The path to the input image.

        Returns:
            saved_class (list): A list of predicted class labels.
            saved_bbox (list): A list of bounding boxes.
        """
        results = self.yolo_model(img_path)
    
        saved_class = []
        saved_bbox = []

        for pred in results.xyxy[0]:
            conf, cls, bbox = pred[4], pred[5], pred[:4]
            saved_class.append(self.yolo_class_names[int(cls.item())])
            saved_bbox.append(bbox.tolist())

        return saved_class, saved_bbox
    
    def send_to_crop(self, saved_bbox, img):
        """
        Sends the image to the cropping function to extract regions of interest.

        Args:
            saved_bbox (list): A list of bounding boxes.
            img (str): The path to the input image.

        Returns:
            bool: True if cropping is successful, False otherwise.
        """
        try:
            if not os.path.exists('features'):
                os.mkdir('features')
            image_path = img
            original_image = Image.open(image_path)
            for i in range(len(saved_bbox)):
                out_img = original_image.crop(saved_bbox[i])
                img_name = image_path.split('/')[-1].split('.')[0] 
                out_img.save('features/'+img_name+str(i)+'.png')
        except:
            return False
        return True
    
    def clear_features(self):
        """
        Clears the temporary features directory.

        Returns:
            bool: True if clearing is successful, False otherwise.
        """
        try:
            for filename in os.listdir('features'):
                if filename.endswith(".txt"):
                    continue
                file_path = os.path.join('features', filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
        except:
            return False
        return True
    
    def get_dominant_color(self, img_path):
        color_thief = ColorThief(img_path)
        dominant_color = color_thief.get_color(quality=1)

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
                (128, 0, 0, "maroon"),
                )

        def nearest_colour( subjects, query ):
            return min( subjects, key = lambda subject: sum( (s - q) ** 2 for s, q in zip( subject, query ) ) )

        nearest_col  = str(nearest_colour( colours, dominant_color )[-1] )
        return nearest_col
    
    def send_to_cnn(self, img_path):
        """
        Sends the image to the CNN model for image classification.

        Args:
            img_path (str): The path to the input image.

        Returns:
            predicted_class_label (str): The predicted class label.
            nearest_col (str): The nearest color to the dominant color in the image.
        """
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = self.modelcnn.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        class_indices = ['OTHER', 'animal','cartoon','chevron','floral','geometry','houndstooth','ikat','letter_numb','plain','polka dot','scales', 'skull','squares','stars','stripes','tribal']

        predicted_class_label = class_indices[int(predicted_class)]

        nearest_col = self.get_dominant_color(img_path)
        
        return predicted_class_label, nearest_col
        
        
        
if __name__ == '__main__':
    model = Model()
    img = 'two.jpg'
    model.run_model(img)