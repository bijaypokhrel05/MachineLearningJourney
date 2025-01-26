# MachineLearningJourney
___
From today onwards, I will make a notes and share ideas what I had learned. The resources and books that I had covered will be mention here.<br></br>

<img src='images/title_image.jpg' width="450" height="250" />

___
## Syllabus to cover

| S.N. | Resources and Books                                                                                                                    | Status |
|------|----------------------------------------------------------------------------------------------------------------------------------------|--------|
| 1.   | [Machine Learning Specialization: Advance Learning Algorithm](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/1) |        |

___
## Projects


| S.N. | Projects Title | Status |
|------|----------------|--------|
| 1.   | Updated soon   |        |

___
## Day 1: Neural Network Basic
Today, I covered basic **Neural network** and its intuition from the Coursera's [Andrew NG course](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/3wO3h/welcome). Here, I gained intuition behind the neural network. <br></br>
**Definition**: A neural network in machine learning is a ***computational model*** inspired by the way **biological neural networks** in the human **brain** process information. Basically, neural network is a type of algorithms that try to **mimic** the brain.

### Basic structure of Neural Network
- **Input Layer:** This layer receives the input data. Each node in the input layer represents a feature of the input data.<br></br>
- **Hidden Layer:** These layers contain neurons that process the input received from the previous layer. The number of hidden layers and neurons per layer is a crucial design choice.<br></br>
- **Output Layer:** The final layer produces the output, such as a predicted value or classification result. The number of neurons in this layer corresponds to the number of possible outputs.<br></br>
- **Activation Function:** The term 'a' stands for activation, and it's actually a term from neuroscience and refers to how much a neuron is sending a high output to other neurons downstream from it. Each neuron in the hidden and output layers uses an activation function to transform the weighted sum of its inputs. Common activation functions include ***ReLU*** (Rectified Linear Unit), ***Sigmoid***, and ***Tanh***. The activation function introduces ***non-linearity***, which is essential for the network to learn complex patterns.

![Demand Predictions](/images/day1_neural_network.png)


### Things to be noted
- When building neural network, one of the decisions you need to make is how many hidden layers do you want and how many hidden layers do you want each hidden layer to have. And this question of how many hidden layers is a question of the ***architecture*** of the neural network.

![Car Predictions](/images/day1_car_predictions.png)

As you can see in above picture that there are more than one hidden layers also called as multiple hidden layers and sometimes refers as ***multilayer perceptron***.Multiple hidden layers enable neural networks to learn complex, ***hierarchical features*** from data, making them powerful for tasks like ***image classification***, ***speech recognition***, and ***language understanding***. The depth of a neural network significantly enhances its ability to model complex patterns, but it also introduces challenges like ***overfitting***, requiring careful training and ***regularization*** techniques.
___

## Day 2: Neural Network Model
I gained how neural network model are design and understand some important notation that needed to know while working with neural network.

![Neurons and Layers](/images/neural-network_day2.png)

- In above picture, ***logistics regression*** as the activation function in a neural network is essentially using the ***sigmoid*** function as the activation function for neurons, particularly in binary classification problems.
<br></br>
- The picture only contains one *hidden layer* and hidden layers has three *neurons*. The output layer contains only one neuron and output of the hidden layer is the input of *output layer*. `a` is the vector of activation values from layer 1.

### Notation under Neural Network
![Notation](/images/network-layer_day2.png)

- Let's look up! The thing to remember is whenever you see this ***superscript square bracket 1*** i.e `a^[1]`, that just refer to the quality that is associated with layer **1** of the neural network. and if you see ***superscript square bracket 2*** i.e `a^[2]`, that refers to a quality associated with layer **2** of the neural network and similarly for other layers as well.<br></br>

- As you can see, `a^[1]` becomes the input to layer **2**. So, the input to layer **2** is the output of layer **1**. Generally, the ***input features*** will be the ***output vector*** from the previous layer.<br></br>
- In above, there are 4 layers and `x` vector is input to neural network and not consider as an ***activation vector***. If we assume that `x` is an activation of layer **0** i.e. `a^[0]`. Then we can generalize that ***output of layer*** `l-1` (i.e. *previous layer*)<br></br>
- Parameters `w` and `b` of layer `l`, unit `j`. In each particular layers, number of neurons (unit) contained and denoted as ***subscript j***. It's just like rows of matrix. And we denoted `j` for each neuron associate with `l` layer and `l-1` layer's **activation function** is the input of layer `l`.

![Handwritten digit recognition](/images/handwritten-day2.png)

I go through this [Notebook](/lab_session_1) and learn some *Tensorflow and Keras* stuff and understand how logistics and neural network distinguish.

___
## Day 3: Build the model using Tensorflow

> *"One of the remarkable things about neural network is the same algorithm can be applied to so many different application."*

Today, I look about how tensorflow and numpy array distinguish. Inorder to build the model let's first understand how tensorflow and numpy works.

![Day 3 tensorflow](/images/day_3_note_tf_np.png)

- Numpy was first created and becomes a standard library for linear algebra in Python.<br>
- In above, we can see that matrices can be represented in rows and columns form. And numpy array can help to achieve representation of matrix.<br>
- This numpy array uses broadcasting, slicing which makes efficient for computation of matrix.<br>
- Tensorflow and Numpy should be used wisely because they contain similar type of workflow and makes us illusion.

![Day 3 tensorflow 1](/images/day_3_tensorflow1.png)

![Day 3 tensorflow 2](/images/day_3_activation_vector.png)
**Tensorflow** is a machine learning package developed by Google. In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0. Keras is a framework developed independently by François Chollet that creates a simple, layer-centric interface to Tensorflow.

- In order to use the Dense object we should import some packages.
```python
from tensorflow.keras import Dense
```

- ***Dense*** is another name for the layers of a neural network that we've learned about so far. As we learn more about neural network, we learn about other types of layers as well.<br></br>
- A **tensor** here is a datatype that the **Tensorflow** team had created inorder to store and carry out computational on matrices efficiently. So, whenever you see tensor just think of the matrix on these above images.Technically, a tensor is a little bit more general than the matrix.<br></br>
- We can convert the tensor into numpy as you can see in  below code:
```python
x = np.array([[200.0, 17.0]])
layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x)

# We can convert the tensor object into teh numpy
a1.numpy()   # Output: array([[0.2, 0.7, 0.3]], dtype=float32)
```
- Return it in the form of numpy rather than in the form of ***tensorflow array*** or ***Tensorflow matrix***.

So, we collect some of the basic information. We are ready to discuss the model implementation. See the code of *digit classification model* in picture below:

![Full implementation](/images/day_3_full_implementation.png)

- Firstly, created a layers that we needed to design a model and using ***sequential*** object of Tensorflow, ***compile*** we can fit and predict easily. However, I haven't explained details of above mention objects and will understand in further session. 

[Notebook of Today](/week_1_lab_session/C2_W1_Lab02_CoffeeRoasting_TF.ipynb)
___
## Day 4: Implementation of Forward Propagation from Scratch

Today, from Coursera's [Machine Learning Specialization](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/1) I had explored the important concept and after that I dived into the scratch implementation of neural network. All the coding stuff and core concepts from today session are mention below:

- **Core Concept:** ***Forward*** and ***backward propagation*** are the core mechanics that allow the network to learn from data and improve its performance over time.
    - **Forward propagation** is where the network makes a guess based on input.
    - **Backward Propagation** is where the network learns from its mistake by adjusting how it makes those guesses.<br></br>
- These steps happen over and over, and each time, the network gets better at making accurate predictions. However, I had implemented ***forward prop*** only for today and another will be covering later on this specialization.

![Day 4 scratch implementation](/images/Day_4_from_scratch.png)

The picture shows how we can actually implement neural network using numpy only but *quite lengthier*. Although, we don't perform like this in ***production level***. There are different framework such as ***tensorflow*** and ***pytorch***. These ***frameworks*** make our works easier. But to be a good machine learning engineer it's a good practice to have scratch implementation of any algorithm so that can help us in ***debugging***.

![Day 4 Forward prop](/images/day_4_forward_prop.png)

To make task easier, I had mentioned code below:
```python
import numpy as np

# Let's define sigmoid function
def g(z):
  return  1/(1+np.exp(-z))

# Also define a dense function
def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out = g(z)
    return a_out

# Definition of sequential function
def sequential(x):
    a1 = dense(x, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    f_x = a4
    return f_x
```

### AGI: Artificial General Intelligence
![Day 4](/images/day_4_agi_ani.png)

I feel AI is superior but looking above makes broaden my thinking. **AGI** stands for ***Artificial General Intelligence*** which is much more complex as human brain. In today, we only focus to solve one particular problem but if we can generalize it (i.e. *can solve any type of problem like human*). This concept fascinates me and will learn more about that in upcoming days.

By learning this much, I had completed my *week 1* from [Advance Learning Algorithm](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/1) course. And from tomorrow onwards, I will enter to *week 2* learning journey. Stay tuned!
___
## Day 5: 







