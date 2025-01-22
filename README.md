# MachineLearningJourney
___
From today onwards, I will make a notes and share ideas what I had learned. The resources and books that I had covered will be mentioned here.<br>

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
## Day 1
Today, I covered basic **Neural network** and its intuition from the Coursera's [Andrew NG course](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/3wO3h/welcome). Here, I gained intuition behind the neural network. <br></br>
**Definition**: A neural network in machine learning is a ***computational model*** inspired by the way **biological neural networks** in the human **brain** process information. Basically, neural network is a type of algorithms that try to **mimic** the brain.

### Basic structure of Neural Network
- **Input Layer:** This layer receives the input data. Each node in the input layer represents a feature of the input data.
- **Hidden Layer:** These layers contain neurons that process the input received from the previous layer. The number of hidden layers and neurons per layer is a crucial design choice.
- **Output Layer:** The final layer produces the output, such as a predicted value or classification result. The number of neurons in this layer corresponds to the number of possible outputs.
- **Activation Function:** The term 'a' stands for activation, and it's actually a term from neuroscience and refers to how much a neuron is sending a high output to other neurons downstream from it. Each neuron in the hidden and output layers uses an activation function to transform the weighted sum of its inputs. Common activation functions include ***ReLU*** (Rectified Linear Unit), ***Sigmoid***, and ***Tanh***. The activation function introduces ***non-linearity***, which is essential for the network to learn complex patterns.

![Demand Predictions](/images/day1_neural_network.png)

### Things to be noted
- When building neural network, one of the decisions you need to make is how many hidden layers do you want and how many hidden layers do you want each hidden layer to have. And this question of how many hidden layers is a question of the ***architecture*** of the neural network.

![Car Predictions](/images/day1_car_predictions.png)

As you can see in above picture that there are more than one hidden layers also called as multiple hidden layers and sometimes refers as ***multilayer perceptron***.Multiple hidden layers enable neural networks to learn complex, ***hierarchical features*** from data, making them powerful for tasks like ***image classification***, ***speech recognition***, and ***language understanding***. The depth of a neural network significantly enhances its ability to model complex patterns, but it also introduces challenges like ***overfitting***, requiring careful training and ***regularization*** techniques.
___