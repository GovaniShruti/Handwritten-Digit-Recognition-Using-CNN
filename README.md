# Handwritten-Digit-Recognition-Using-CNN
<p>Basically, in this project, I have developed a deep learning model which can recognize handwritten digits.</p>
<h3>Description</h3><hr>
<p> In the era of digitization, Handwritten digit recognition plays vital role in information processing. It is the ability of machines and computers to identify handwritten digits. By implementing this method machines can recognize the digits present in the image. These systems are widely utilized in a variety of sectors, but the variance and distortion of handwritten character sets pose a significant difficulty for computers to recognize digits because different communities may use different styles of writing. As a result, one of the most important tasks in the digit recognition system is digit identification using the best distinguishing features. There are various approaches available to recognize handwritten digits based on machine learning and deep learning techniques. Several algorithms such as K-Nearest Neighbors, Support vector machine, Convolution neural network are used for it. These classifiers are trained with dataset and then used to process any digital scan document into computer document format.I am going to implement a handwritten digit recognition model using the MNIST dataset. MNIST dataset is a predefined dataset, used for training and testing procedure. Here, I introduce a special type of deep neural network that is Convolution Neural Network. Here, my goal is to achieve higher accuracy along with reduced cost and complexity with the help of CNN architecture.
</p>
<p>
Firstly, data preparation is the most important part. So, I have done data preprocessing procedure such as loading data, finding missing values, normalization, reshaping, label encoding and so on. For label encoding, I have encoded labels to one hot vectors. After that in data wrangling step, I have divided the dataset into training and validation sets. In this experiment I have used RMSProp optimizer rather than using Stochastic Gradient Decent (SGD), as SGD is slower than RMSprop. I have used convolutional neural networks for feature extraction and classification. Using this model, I got 99.6% training accuracy by using 30 epochs. </p>
<h3>Methodology Flowchart</h3><hr>![flowchart](https://user-images.githubusercontent.com/88697274/129999801-febfa610-4f9f-4a06-9c30-4f34c87662cb.jpeg)


<h3>Getting Started</h3><hr>
<b>Dependencies</b>
 <ul>
<li>Pandas</li>
<li>Numpy</li>
<li>Keras</li>
<li>Tensorflow</li>
<li>Matplotlib</li>
</ul>
<b>Language Used</b>
 <ul>
<li>Python : version above 3.0</li>
</ul>
<h3>Dataset Description</h3><hr>
I am going to implement a handwritten digit recognition model using the MNIST dataset. MNIST dataset is a predefined dataset, used for training and testing procedure. There are 60,000 training images and10,000 testing images, images from MNIST were normalized to fit into a 28*28 pixel bounding box and anti-aliased, that introduces grayscale levels. And it can be downloaded from <a href="https://www.tensorflow.org/datasets/catalog/mnist">here</a>
<h3>Manifest</h3><hr>
Final_Research_Project.ipynb / Final_Project.py :
<div>
  <ul>
    <li>Code for Image Preprocessing</li>
    <li>Code for CNN Model</li>
    <li>Code to train and validate model</li>
    <li>Code to plot the results</li>
  </ul>
  </div>
<h3>Executing program</h3><hr>
<p>Downloaded dataset from given link. <br>
If using local machine download libraries given in dependencies. <br>
If the running environment is google colab just make a folder on the drive named "HDR" and put the downloaded dataset into it. </p>
<ul><li>Method 1 (For colab):<br>Download and run Final_Research_Project.ipynb file which will execute the program.</li>
<li>Method 2 (For local machine):<br>
Run Final_Research_Project.ipynb to execute the whole project in the local machine. Make sure you change all the paths given in code according to your local machine.</li></ul>
<h3>Results</h3><hr>
<p>Accuracy for training and validation set and Loss for training and validation dataset can be found in Results folder.</p>
<h3>Authors</h3><hr>
Shruti Govani <br>
Feel free to contact me at <a href="mailto:sgovani@lakeheadu.ca">sgovani@lakeheadu.ca</a>
<h3>License</h3><hr>
This Project is not Licensed
<h3>Project Status</h3><h
r>
Project is completed but improvements can be made in terms of accuracy.
