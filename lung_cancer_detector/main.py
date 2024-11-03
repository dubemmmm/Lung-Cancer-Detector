import base64

from flask import Flask, request, render_template, url_for
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# This variable will store the received image data
received_image = None


@app.route('/')
def home():
    return render_template('predict.page.html')


@app.route('/predict', methods=['POST'])
def upload_image():
    returnedResponse = {}
    try:
        global received_image
        print(request.form)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        termsAndCondition = request.form["termsAndCondition"]

        sex = request.form['sex']
        print(termsAndCondition)

        if sex == "No option":
            returnedResponse['error'] = "Pick a gender option"
            return render_template("predict.page.html", returnedResponse=returnedResponse)

        if 'lungPic' in request.files:
            # Load saved model
            model = tf.keras.models.load_model("best_lung_cancer_model.h5")

            # Get uploaded image and convert to tensors
            image = request.files['lungPic']
            image = image.read()

            encoded_image = base64.b64encode(image).decode('utf-8')
            returnedResponse["imageBase64"] = encoded_image

            image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # Preprocess image
            grayscale_image = np.mean(image[:, :, :3], axis=2)
            grayscale_image = np.expand_dims(grayscale_image, axis=2)
            resized_image = tf.image.resize(grayscale_image, size=(28, 28))
            print(resized_image.shape)
            img = np.array([resized_image])
            print(img.shape)

            # Perform predictions
            pred = model.predict(img)
            print(pred)
            # Highest probability
            maxIndex = np.argmax(pred[0])

            # Set the returned description
            if maxIndex == 0:
                returnedResponse["description"] = "This is a cancerous lung"
                returnedResponse["cancer"] = True
            else:
                returnedResponse["description"] = "This lung is not cancerous"
                returnedResponse["cancer"] = False

            # Get the class labels
            class_labels = ["Malignant", "Normal"]
            result = class_labels[maxIndex]

        else:
            return 'No image received in the request.', 400


        returnedResponse["fullname"] = firstname + " " + lastname
        return render_template("result.page.html", returnedResponse=returnedResponse)

    except KeyError:
        returnedResponse['error'] = "Please accept terms and conditions"
        return render_template("predict.page.html", returnedResponse=returnedResponse)


if __name__ == '__main__':
    app.run(debug=True)
