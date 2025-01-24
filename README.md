# MachineLearningJourney
___
From today onwards, I will make a notes and share ideas what I had learned. The resources and books that I had covered will be mentioned here.<br></br>

<img src='images/title_image.jpg' width="450" height="250" />

___
## Syllabus to cover

| S.N. | Resources and Books                                                                                   | Status |
|------|-------------------------------------------------------------------------------------------------------|--------|
| 1.   | [Advance Learning Algorithm](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/1) |        |

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
- Parameters `w` and `b` of layer `l`, unit `j`. In each particular layers, number of neurons (unit) contained. It's just like rows of matrix. And we denoted `j` for each neuron associate with `l` layer.

![Handwritten digit recognition](/images/handwritten-day2.png)

I go through this [Notebook](/lab_session_1) and learn some *pytorch* stuff and understand how logistics and neural network distinguish.

___
## Day 3: 

