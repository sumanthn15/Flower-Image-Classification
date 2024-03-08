# Flower Image Classification

This is a simple project that uses a convolutional neural network (CNN) to classify images of flowers into different categories. The project consists of two Python files:

- `model_trainer.py`: This file contains the code to train the CNN model using the flower images.
- `app.py`: This file contains a GUI application that allows the user to select an image of a flower and get a prediction of the flower's category.

## Dataset

The dataset used in this project contains images of five different types of flowers:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

The images are stored in the `flowers` directory, which is organized into subdirectories for each flower type.

## Training the Model

To train the model, simply run the `model_trainer.py` file. The script will load the images from the `flowers` directory, preprocess them, and train the CNN model using TensorFlow. The trained model will be saved to a file named `flower_classifier`.

## Running the Application

To run the GUI application, simply run the `app.py` file. The application will open a window that allows the user to select an image of a flower. Once the user selects an image, the application will preprocess the image, make a prediction using the trained CNN model, and display the predicted flower category.

## Requirements

This project requires the following Python packages:

- TensorFlow
- Pillow
- tkinter

To install these packages, you can use pip:

pip install tensorflow pillow tkinter


## Conclusion

This is a simple project that demonstrates how to use a CNN to classify images of flowers. The project can be easily extended to include other types of images or to improve the accuracy of the model.
