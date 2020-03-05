# Leggo My Lego
### A Lego Mini Figure Detection Android App

## (Computer) Vision
I have been interested in computer vision for a long time because of its huge potential in robotics, self-driving cars, manufacturing, and many other places in life. Detecting Lego mini figures was an object detection project with a real business application small enough to be completed in the week and a half allotted. A company I am working with has plans to make an app that will be able to catalog an entire Lego collection, and this is one step in that direction.


## Training Data Generation
Training data for object detection requires images and bounding boxes. Bounding boxes for training are usually made by hand, which is a very time-consuming process, but I was able to generate rendered data in such a way that it also generated the bounding boxes. My data generation pipeline was:
- Generate 3D mini figures in MLCad
- Use Quick Macros to automate control of MLCad
- Convert models to POV-Ray using LDRaw Viewer
- Auto-modify the models using Python
- Render the models using POV-Ray
- Post processing and get bounding boxes with python

Most of the data generation functions are in src/datagen_functions.py and src/background_placement.py. Conversion to tensorflow shards is in src/tf_dataset_conversion.py<br>
Here is an example of my data:

<img src="assets/random_0122-r1.png" alt="alt text" width="300">

In a future version, training data will be augmented by photos from Lego of their official minifigures.

## Model

I used transfer learning to get a neural network to detect the mini figures. Neural networks are complex algorithms that take inspiration from human brains and the neurons that make them up. Transfer learning is a technique that also imitates the way human brains work, specifically our ability to learn what a new object looks like without having to re-learn how to see at all.

## Mobile App

I built an android app that can be found at dtothe3.com/leggomylego 

<img src="assets/Screenshot_1.png" alt="alt text" width="250">
