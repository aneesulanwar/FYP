import flask
import werkzeug
import shutil
import PredictYolo
import cv2
import CustomFunctions
from keras import backend as K
from flask import jsonify, send_file
from keras.models import load_model
import numpy as np
import tensorflow as tf

app = flask.Flask(__name__)

label = "0"

graph = tf.get_default_graph()
testdict = {'cropped_ptcl_address': 'SYED MUJEEB ALLAH HUSSAIN\nH 87\nST 486\n\nG--13/1, IBA\n1S] AMARARD', 'cropped_ptcl_dateblock': 'Billing Month Nov. 2015\n\n \n\n \n\nAmount After Due Date Rs. 2,690.00', 'cropped_ptcl_grandtotal': 'Grand Total Rs. 2,560.00', 'cropped_ptcl_header': 'rakistan Telecommunication\nCompany Limited 4', 'cropped_ptcl_phonenumber': '|_dateblock /1.14%\n051-2323158'}

testdict1 = {'cropped_gAddress': 'Name: a HANIFA BEGUM\n\nAddress: 5% HINO 316-17-18\n\nAHATA KHAZANCHI\n\nSADDAR BAZAR\n\nRAWALPINDI\n246607337258\n\n7\n\nBill ID\n\nPu grocuss cin 1 area ee', 'cropped_gMeter': 'MR01599154\n\nPeriod', 'cropped_gUnits': 'Difference 3)\n00161000 _\n\nsgt eiceaee Oa aaaas', 'gData2.png': '‘| Dec 2017\n[4,760\n\n(14-04-2018\nno\n\nPee\nETT ee'}

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)

    imagefile.save(filename)
    img2 = cv2.imread("androidFlask.jpg")
    print("Shape ",img2.shape)

    # Copying saved image to a new folder
    shutil.copy("androidFlask.jpg", "ptcl_test_image/")

    # Executing the Yolo Test Model
    label = exec_classifier()
    text_ocr = PredictYolo.executePredictYolo(label)  # exec_classifier() returns class of image i-e 1 or 2 or 3

    print("ocr text: ", text_ocr)
    print("type is :", type(text_ocr))

    text_ocr=CustomFunctions.processText(text_ocr,"2")

    print(type(text_ocr))

    # Transfer Outputs to OutputHistory Folder
    dest_directory = CustomFunctions.createNewDir("OutputsHistory/")
    CustomFunctions.moveFilesToNewDir("Outputs", dest_directory)

    text_with_image = dest_directory + "/androidFlask.jpg"
    base64_img = CustomFunctions.getImageBytes(text_with_image)


    # Clearing the session removes all the nodes left over from previous models, freeing memory and preventing slowdown.
    # Done to handle multiple concurrent requests of Client

    K.clear_session()

    return jsonify(text_ocr)


def exec_classifier(): # function for classification of image
    width = 64
    height = 64

    image = cv2.imread("androidFlask.jpg")
    #output = image.copy()
    image = cv2.resize(image, (width, height))
    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0

    # when working with a CNN: don't flatten the image, simply add the batch dimension
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    model = load_model("first_CNN_model")
    preds = model.predict(image)

    # find the class label index with the largest corresponding probability
    probas = np.array(preds)
    labels = np.argmax(probas, axis=-1)
    labels = str(labels[0])
    print("label is ", labels)
    return labels

def test_image():
    img2 = cv2.imread("androidFlask.jpg")

    # show image received
    # cv2.imshow('imagetrest',img2)
    # cv2.waitKey()

    # Copying saved image to a new folder
    shutil.copy("androidFlask.jpg", "ptcl_test_image/")

    # Executing the Yolo Test Model
    text_ocr = PredictYolo.executePredictYolo(exec_classifier()) #exec_classifier() returns class of image i-e 1 or 2 or 3
    print("ocr text: ",text_ocr)
    print("type is :",type(text_ocr))
    #res_dict = CustomFunctions.listToString(text_ocr)
    text_ocr=CustomFunctions.processText(text_ocr,"0")
    # Transfer Outputs to OutputHistory Folder
    dest_directory = CustomFunctions.createNewDir("OutputsHistory/")
    CustomFunctions.moveFilesToNewDir("Outputs", dest_directory)

    text_with_image = dest_directory + "/androidFlask.jpg"
    base64_img = CustomFunctions.getImageBytes(text_with_image)
    # text_ocr[10]=base64_img
    # Clearing the session removes all the nodes left over from previous models, freeing memory and preventing slowdown.
    # Done to handle multiple concurrent requests of Client
    #print(text_ocr)
    K.clear_session()

    return jsonify(text_ocr)


if __name__ == "__main__":
    # print("hello")

    with app.app_context():
        #getjson=test_image()
        #app.run(host="192.168.18.65", port=5000, debug=True)
        #exec_classifier()
        #print(getjson)
        CustomFunctions.processText(testdict1,"0")

# To specify the socket of the server:
# (IPv4 address and port number to which the server listens for requests), the run() method is used

# Inside the handle_request() function, the uploaded image can be retrieved from the MultipartBody form
# sent from the Android app.
# The uploaded files are stored into the 'files' dictionary, which can be accessed using flask.request.files.
# The filename specified in the Android app, which is 'image',
# will be used to index this dictionary and return the uploaded image file.

# preferred way to return the file name associated with the file is to use the werkzeug.utils.secure_filename() method.
# The file name will be printed and then can be saved using the save() function.
# The file will be saved in the current directory.
# Finally, a message is sent back to the Android app using the return statement.
# python -m pip install scipy on cmd to install a lib


