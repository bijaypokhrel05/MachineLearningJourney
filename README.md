# MachineLearningJourney
___
From today onwards, I will make a notes and share ideas what I had learned. The resources and books that I had covered will be mention here.<br></br>

![Title image](/images/title_images.jpg)

___
## Syllabus to cover

| S.N. | Resources                                                                                                                                                                                    | Status                                                           |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 1.   | [**Machine Learning Specialization: Advance Learning Algorithm**](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/1)                                                   | ![✔️](https://img.shields.io/badge/Status-Completed-brightgreen) |
| 2.   | [**Machine Learning Specialization: Unsupervised Learning, Recommenders, Reinforcement Learning**](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning) |  ![✔️](https://img.shields.io/badge/Status-Completed-brightgreen)                                                                |
| 3.   | [**Neural Network Playlist from 3Blue1Brown**](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)                                                          | ![✔️](https://img.shields.io/badge/Status-Completed-brightgreen) |

___
## Projects


| S.N. | Projects Title                                                                         | Status |
|------|----------------------------------------------------------------------------------------|--------|
| 1.   | [Email/SMS spam detection](https://github.com/bijaypokhrel05/Email-SMS-Spam-Detection) |   ![✔️](https://img.shields.io/badge/Status-Completed-brightgreen)     |

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
## Day 5: Train Neural Network in Tensorflow
In this specialization, I dived into *week 2* with the full implementation of neural network in **Tensorflow**. Again taking the **digit classification problem** and with ***systematic 3 steps***, we conclude our today updates.

### 1. Create the model
![Day 5 step 1](/images/Day_5_step2.png)

Here, we define a model and import all the **tensorflow dependencies** as mention in above code. ***Sequential and Dense*** from *keras* will be helpful for efficient execution of program.

### 2. Loss and Cost function
![Day 5 step 2](/images/Day_5_step3.png)

**Loss** can be different depending upon what type of problem you have taken. When you work on predicting numbers, then ***loss** will be ***mean squared error*** and similarly ***binary cross-entropy*** (also known as ***logistic loss***) is a way to measure how "wrong" or "right" the model was in its prediction, based on the actual truth (whether the email is actually spam or not).

### 3. Gradient descent
![Day 4 step 3](/images/Day_5_step1.png)

Visualizing Loss function `J(w)` over weights `w` and compute the minimum value using partial derivatives concept. Where ***alpha*** should be taken in such a way that weight and bias for higher order must be smaller. ***Epochs*** is the number of steps in gradient descent.

### Full implementation using Tensorflow
![Day 4 complete implementation](/images/Day_5_complete_implementation.png)
___
## Day 6: Types of activation, softmax regression, multi-label classification

**Activation function** plays a crucial role during modeling neural network. Activation function are heart of most machine learning model. Today, I dived into some of the important types of activation functions. 
They are explained below:

![Day 6 Type of activation](/images/Day_6_activation_func.png)
![Day 6 activation](/images/DAy_6_activation_depth.png)
- **Linear activation function** are the most general activation. Some people assume no use of activation during use of linear activation function. When we use to predict any numbers either -ve or +ve, linear activation function is useful.<br></br>
- **ReLU(Rectified Linear Unit)** is the another most common activation function. Basically,when we need to predict the house price we use ReLU because price can't be in negative `ReLU = max(0, z)`. It is very efficient because of its simplicity and mostly used for the hidden layers.<br></br>
- **Sigmoid** is the another popular activation function basically deals with the binary classification type of problem. Its discrete value ranges between 0 and 1. However, it is not suitable for the multi class classification problems such as handwritten digits recognition problem. For the multi class classification problem we have to use the softmax activation function because it is the generalize form of sigmoid function. Comparison between the logistics and softmax  and also cost difference is shown in below images.

![Day 6 softmax implementation](/images/Day_6_logistics_softmax_cmpr.png)
![Day 6 softmax logistic cost](/images/Day_6_softmax_log_cost.png)

### Multi-class classification and softmax
**Multi-class classification** is a type of machine learning problem where the goal is to categorize data into one of more than two possible classes. For example, classifying an image as either a dog, cat, or bird is a multiclass problem because there are multiple categories (classes) to choose from.

**Softmax** is a _mathematical_ _function_ that helps solve this by turning the output of a model (which could be raw scores or **_logits_**) into probabilities. It does this by taking the scores and exponent them, and then _**normalizing**_ them so that the sum of all **_probabilities equals 1_**. This way, each class has a **_probability_**, and the class with the highest probability is chosen as the model’s prediction.

**In short**: **_Softmax_** helps in multiclass classification by converting model outputs into understandable **_probabilities_** for each class. The class with the highest probability is selected as the predicted class.

![Day 6 MNIST with softmax](/images/Day_6_mnits_with_softmax.png)
The above image show the full implementation of neural network with the **MNIST** datasets using softmax. However, **Andrew Ng** doesn't recommend us to use it because of not so accurate decimal values. Instead use it after certain modification is done. Some of the modification is highlighted in below image.

![Day 6](/images/Day_6_complete_accurate_implementation.png)
Some changes are output layer's activation changes to ***linear*** and instead of using loss as ***binarycrossentropy***, we use ***SparseCategoricalCrossentropy***. These modification helps to achieve with the more numerically accurate results.

___
## Day 7: Convolution Layer, Backward propagation
### Additional Neural Network concept
In ***Dense layer*** , each neuron output is a function of all the activation outputs of the previous layer. While in case of ***Convolution layer***, each neuron only looks at part of the previous layer's output.

![Day 7 convolution layers](/images/Day_7_convolution_layer.png)
### Why convolution layers?
- **Faster computation**
- **Need less training data**
- **Less prone to overfitting**

### Gradient Descent - Backward propagation
![Day 7 backward propagation](/images/Day_7_backward_prop.png)
Gradient descent requires the derivative of the cost with respect to each parameter in the network. Neural networks can have millions or even billions of parameters. The back propagation algorithm is used to compute those derivatives. Computation graphs are used to simplify the operation. 

#### Computation Graph
A computation graph simplifies the computation of complex derivatives by breaking them into smaller steps. And if you want to visualize more deep into the backward propagation, then this graph concept provide you with better intuition.

In this *week 2*, I had learned about the different type of activation function, training details and multi-class classification, backward propagation details. And from tomorrow onward, I will jump to *week 3* on Coursera's [Advance Learning Algorithm](https://www.coursera.org/learn/advanced-learning-algorithms/home/week/2).

___
## Day 8: Evaluating a Model
With short learning, I started a new *week 3* session from today. I learnt about how we can evaluate our model. Due to imbalance datasets, it is very difficult to generalize our model to new data. However, we manage low biases during the training process. But cross validation and `KFold` helps us to reduce the overfitting problem.

### Diagnostic:
A test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance.

![Day 8 evaluating model](/images/Day_8_evalutating_model.png)

In above, to determine the mean squared error metric we will not include the regularization term.

![Day 8 model selection](/images/Day_8_model_selection.png)
All the details are shown in the above picture, we can easily understand.

![Day 8 choosing model](/images/Day_8_choosing_model.png)

![Day 8 performance](/images/Day_8_performance.png)

___
## Day 9: Bias and Variance
Today, I learnt about **bias** and **variance** concept which is a crucial part of the machine learning cycle. In machine learning, **bias** and **variance** are two sources of error that affect how well your model performs. _The goal is to find the right balance_.

![Day 9 bias variance](/images/Day_9_bias_variance.png)

* **Too much bias**: Your model is underfitting and missing important patterns.
* **Too much variance**: Your model is overfitting and capturing noise instead of the true signal.

By understanding bias and variance, you **_can tweak your model_** to make it more **_accurate_** and **_reliable_**!
___
## Day 10: Bias and variance with varying lambda, Learning Curve
Today, I delved into understanding the fundamental that every ML engineer should know. While varying _regularization parameter_ i.e. ***lambda***, what differences can occur in bias and variance.

![Day 10 bias and variance cv](/images/Day_10_bias_variance_while_lambda_changes.png)
- In above figure, **CV** stands for _**cross validation**_ is a sampling technique. It helps in estimating how well the model will perform on an independent dataset.
- Also, shows the two curves which are exactly mirror of both. They show that while varying ***lambda*** and ***degree of polynomial***, how loss function of **train** and **cv** makes their behavior.

![Day 10 bias and variance](/images/Day_10_bias_variance_while_lambda_changes1.png)
- As we can see that when lambda is taken as larger value, the algorithm is highly motivated to keep these `w` very small and so you end up with `w1`, `w2` and really all parameters will be very close to zero.<br></br>

- Similarly, when lambda is taken small, it means no regularization term. So we're just fitting the fourth order polynomial. We end up with that curve that you saw in the above picture. Then model will have high variance (**overfit**) because it fit all the training data and fail to generalize new data.<br></br>
- And in middle of the above picture, is the more generalize case and have intermediate _**lambda**_ value. Where loss function of both **cv** and **train** has smaller magnitude.

### Learning Curve
![Day 10 learning curve](/images/Day_10_learning_curves.png)

**Learning curves** helps in diagnosing underfit and overfit in machine learning model. It is a graphical representation of how a model's performance changes over time as it learns from more data or as the training progresses. Let's dive in graphical views of **high bias** and **high variance** cases:
- **High bias**

![Day 10 learning curve with high bias](/images/Day_10_learning_curve_with_high_bias.png)

- **High variance**

![Day 10 learning curve with high variance](/images/Day_10_learning_curves_with_high_variance.png)
- A **high training loss** indicates the model doesn't fit the data correctly.
- A **high testing loss** indicates the model doesn't generalize well.
___
## Day 11: Neural Network's Bias and Variance, ML Development Process and Data Augmentation
> Someone said that "*After a lot of work experience in a few different companies, he realized that bias and variance is one of those concepts that takes a short time to learn, but takes a lifetime to master*".

### Neural Network and Bias-Variance Tradeoff
![Day 11 bias variance tradeoff](/images/Day_11_bias_variance_tradeoff.png)
In above, we have to balance the complexity that is the **degree of polynomial** or **regularization parameter lambda**. But it turns out that neural network offer us a way out of this dilemma of having tradeoff bias and variance with some caveats.
- **When you increase the complexity of a neural network** (e.g., adding more layers or neurons), **bias typically decreases** (the model becomes more capable of capturing complex patterns), but **variance increases** (the model becomes more prone to overfitting).<br></br>

- **When you decrease the complexity of a neural network, bias increases** (the model becomes too simple to capture complex patterns), but **variance decreases** (the model becomes less sensitive to the training data and more robust to noise).<br></br>

![Day 11 neural network bias - variance](/images/Day_11_neural_network_bias_variance.png)
- Large neural network are low bias machines. This above images show the clear ideas behind the neural network's bias and variance. 

### Machine Learning Development Process
![Day 11 ml development process](/images/Day_11_ml_development_process.png)

The iterative nature of machine learning development revolves around **trial** and **error**, **continuous learning, and improvement**. Through experimentation, **error analysis**, and **constant refinements**, machine learning models can be progressively optimized. This iterative approach, combined with **solid data** and **model evaluation**, is central to building successful machine learning applications.


### Data Augmentation
![Day 11 data augmentation](/images/Day_11_data_augmentation.png)

Beyond getting brand new training example `(x, y)`, there's another technique that's widely used especially for images and audio data that can increase your training set size significantly called as **data augmentation**. Creating additional examples like this hold the learning algorithm, do a better job learning how to recognize the letter A in above figure. Learnt some of the important topics related to data augmentation:

- **Data augmentation by introducing distortions**.
- **Data augmentation for speech recognition**.

___
## Day 12: Synthesis Data, Engineering Data, Deployment, Classification Metrics

Today, I learnt about the **synthesis data**. **Synthesis data** are the artificial data develop to increase the performance of the model. Basically, using artificial data inputs to create a new training example. All other learning updates are mention below:

### Engineering Data
Most machine learning researchers attention was on the **_conventional model centric approach_**. You can take the reference in below image:

![Day 12 engineering data](/images/Day_12_engineering_data.png)

A **machine learning system** or an **AI system** includes both code to implement your model, the data that you train the algorithm model as well. And over the last few decades, most researchers doing **_machine learning research would download the dataset and hold the data fixed_** while they focus on improving the code or the algorithm or the model. Sometimes it can be more fruitful to spend more of your time taking a data centric approach in which you focus on engineering the data used by your algorithm.


### Full cycle of a machine learning project
![Day 12 ML cycle](/images/Day_12_ML_cycle.png)

The figure illustrates the **full cycle of a machine learning project**, which consists of four key stages:

1. **Scope Project (Define project)**:  
   - This step involves defining the problem you are trying to solve and determining the project's objectives.  
   - Key questions addressed here include: What is the goal of the project? What business or research problem are you solving? What will success look like?<br></br>

2. **Collect Data (Define and collect data)**:  
   - In this stage, relevant data is identified, collected, and prepared for use.  
   - Data may be sourced from databases, APIs, manual collection, or other means. Data cleaning and preprocessing (e.g., handling missing values, normalization) are also part of this step.<br></br>

3. **Train Model (Training, error analysis & iterative improvement)**:  
   - The collected data is used to train machine learning models.  
   - Error analysis is performed to evaluate model performance and identify potential improvements.  
   - Iterative refinement and tuning (e.g., hyperparameter optimization, feature engineering) take place to achieve better accuracy and reliability.<br></br>

4. **Deploy in Production (Deploy, monitor, and maintain system)**:  
   - Once the model is ready, it is deployed into a production environment where it serves real-world tasks.  
   - Monitoring and maintaining the deployed model is essential to ensure consistent performance, including adapting to changes in data patterns or requirements.

The arrows between the stages indicate that this cycle is **iterative**—you may need to revisit earlier steps based on results or new requirements (e.g., refining the project scope, collecting additional data, or retraining the model). This ensures continuous improvement and adaptation to changing conditions.

### Deployment
The **deployment model** is the process of integrating a trained machine learning model into a production environment where it can deliver predictions or decisions in real-world applications. Below image show detail about deployment:

![Day 12 deployment model](/images/Day_12_deployment_model.png)

In general term, there is a mediator called as **API** (Application Program Interface) between the inference server and mobile app / website. For the larger scale, we need a software engineer to meet the following objective:
- Ensure reliable and efficient predictions.
- Scaling
- Logging
- System monitoring
- Model updates

### Classification Metrics
When our datasets contain an imbalanced data or **skewed data**, then accuracy fails to give proper idea. When positive labeled data contain 95% and false labeled data contain 5% which makes the **datasets imbalanced**. So, we can further move on other metrics calculation which will be beneficial for the imbalance dataset.
Use metrics that give you a clearer picture, like:
- **Confusion Matrix**: A table that shows how many predictions were correct, and where the model made mistakes.
![Day 12 confusion matrix](/images/Day_12_precision_recall.png)

- **Precision**: How many of the predicted positives were correct?
- **Recall**: How many of the actual positives did the model catch?
![Day 12 precision recall](/images/Day_12_precision_recall_1.png)


- **F1-Score**: A balance of precision and recall. Below image gives us a crystal idea. 
![Day 12 f1 score](/images/Day_12_f1_score.png)


These metrics can help you understand the real performance of your model beyond just accuracy. With these learning I completed my *week 3* also. And will go on learning *week 4* stuff from tomorrow onwards.
___
## Day 13: Decision Tree
Today, heading towards *week 4* of Coursera's [Advance Learning Algoritm](https://www.coursera.org/learn/advanced-learning-algorithms). I learnt about the **Decision Tree** concept. A **Decision Tree** is a supervised machine learning algorithm used for classification and regression tasks. It is a tree-like structure where each **_internal node_** represents a decision based on a feature, each branch represents an outcome of the decision, and each **leaf node** represents a final prediction (class label for classification or a numerical value for regression).

![Day 13 decision tree](/images/Day_13_decision_tree.png)

### Learning Process of Decision Tree
Two most important decision we must consider and those decisions are mention below:
- **Decision 1**: How to choose what features to split on at each node?
- **Decision 2**: When do you stop splitting?
![Day 13 learning process](/images/Day_13_learning_process.png)

Also revisited neural network concept and practice lab session. I will learn deeper from tomorrow on this decision tree topics. 
___
## Day 14: Entropy and Information Gain
**Note:**
> _Programmatically speaking, decision tree are nothing but a giant structure of nested if-else condition._

> _Mathematically speaking, decision tree use hyperplanes which run parallel to any one of the axes to cut your coordinate system into hyper cuboids._

Today, I dived into the important concept that we have to considered while building the **Decision tree** model. 

### Entropy
![Day 14 entropy](/images/Day_14_entropy.png)

Basically, **entropy** is just measure of **_disorder/impurity_**. As you can see that **entropy** has the parabolic curve with open downwards and has maximum value of `H(p1=0.5) = 1` at the middle of the curve. For better intuition, there is the maximum variation of data at middle. And **low entropy** found when `p1 = 0.0 and p1 = 1.0`. It means only one category found on the **low entropy** instances. In above, `p1` denotes as fraction of examples that are positive. Here, **positive** means what you are trying to infer.

### Information Gain
![Day 14 information gain](/images/Day_14_information_gain.png)

Here, each split of training data should calculate entropy. It doesn't give us an **_average idea_** what feature to choose for the better prediction. So, here's **information gain** comes into picture. **Information gain** generally measure reduction of entropy. As we know that **_entropy_** and **_information gain_** are inversely proportional to each other. Information gain is also known as knowledge. More variation of data gives poor knowledge about our dataset.

### One-Hot Encoding
Also explored about the one hot encoding technique which plays vital role when our category contains more than two discrete values. Generally, machine learning frameworks not able to perform on categorical data. So, this encoding technique helps to convert our **_nominal_** data into binary form to distinguish each datapoints easily.

___
## Day 15: Ensemble Tree, Random Forest, XGBoost
Learnt about the ensemble tree concept. Although, we have already a decision tree algorithm then why we need ensemble tree? Here, a single decision tree is highly sensitive to small changes into data. Due to this limitation, ensemble tree algorithm comes into picture. Simply, a multiple collection of decision tree is ensemble tree. This makes our model less sensitive and making our algorithm more robust.
_**The key idea is that an ensemble of trees works better than an individual tree by averaging their predictions (in regression) or using a majority vote (in classification).**_

### Random Forest
**Random Forest** is a powerful tree ensemble method that builds multiple decision trees and combines their predictions to improve **_accuracy and reduce overfitting_**.

![Day 15 randomizing feature](/images/Day_15_randomizing_feature.png)

**_Key Features of Random Forest:_**
* Uses bagging (**_Bootstrap Aggregation_**): Each tree is trained on a random subset of the data.
* Uses feature randomness: Each split in a tree considers a random subset of features.
* Final prediction is made by majority voting (classification) or averaging (regression).
* Handles missing data and high-dimensional datasets well.
* Reduces overfitting compared to a single decision tree.

### XGBoost
**XGBoost** (**_eXtreme Gradient Boosting_**) is a _**high-performance**_, **_scalable tree-based_** algorithm that improves upon traditional **_Gradient Boosting_** by being faster, more accurate, and optimized for large datasets.
- ![Day 15 xgboost_summary](/images/Day_15_xgboot_summary.png)


- ![Day 15 xgboost](/images/Day_15_xgboost_implement.png)

**_Key Advantages of XGBoost_**
*  Fast and efficient (optimized for speed and memory).
* Prevents overfitting (L1 & L2 regularization).
* Handles missing values and large datasets well.
* Works for both classification & regression.

**When to use decision tree and neural network**<br>
All things are included into the images below:

![Day 15 decide when to use](/images/Day_15_decide_when_to_use.png)

After learning all these things, I completed _part 2_ of **Machine Learning Specialization** from the Coursera. And got certificate for completing this course.
Here is the link of my accomplishment: 
[Completion of Advanced Learning Algorithms](https://coursera.org/share/c025e849589bfe7bb13e7add219cfe64).
___
## Day 16: Optimizing Random Forest for Crop Prediction
Today, I worked on optimizing Random Forest for crop prediction while improving training speed. Initially, GridSearchCV took a long time to train, so I applied the following optimizations:

- **Reduced Search Space** – Limited the range of hyperparameters to focus on the most impactful ones.
- **Lowered n_estimators** – Used 50–150 trees instead of 500 to reduce computation.
- **Limited max_depth** – Set maximum depth to 10–20 to prevent overfitting and speed up training.
- **Decreased n_iter in GridSearchCV** – Reduced it to 10 iterations for a faster search.
- **Used n_jobs=-1** – Leveraged all CPU cores for parallel processing.
- **Lowered Cross-Validation (cv=3)** – Reduced the number of folds to minimize training time.

These improvements significantly **_reduced training time_** by 50-70% while still achieving good accuracy.

Here's some of my code snippet in below:

![Day 16 practice](/images/Day_16_practice_rf.png)

![Day 16 rf 2](/images/Day_16_rf_2.png)

![Day 16 rf 3](/images/Day_16_rf_3.png)

See my code, points in my notebook:
[Notebook For Random Forest](/Day_16_practice_rf)
Also I had downloaded the dataset from the Kaggle.

___
## Day 17: Unsupervised Learning
I've been starting new topic which is beyond supervised learning. This course was designed by **Coursera's** co-founder **Andrew Ng** named as _**Machine Learning Specialization: Unsupervised Learning, Recommender system, Reinforcement Learning**_. This course is divided into three weeks. 
### Topics to cover
- **Unsupervised Learning**
   - _**Clustering**_
   - _**Anomaly detection**_
- **Recommender Systems**
- **Reinforcement Learning**

However, I started Unsupervised Learning. **Unsupervised learning** is a type of machine learning where a model is **_trained on unlabeled data to discover hidden patterns, structures, or relationships_**. Unlike **supervised learning**, there are no **_predefined labels or target variables_**; the algorithm must infer the structure of the data on its own.

### Key Characteristics of Unsupervised Learning
* **No labeled data**: The model learns patterns without explicit supervision.
* **Finds hidden structures**: Clusters similar data points or reduces dimensionality.
* **Exploratory**: Used for understanding data distributions and trends.
* **Commonly used for feature engineering**: Helps create meaningful features for supervised learning.


<center><h3>Clustering</h3></center>

![Day 17 clustering](/images/Day_17_clustering.png)

**Clustering** is a fundamental technique in **unsupervised learning**, where the goal is to group data points into clusters based on their **similarities**. Since clustering is **unsupervised**, there are **no labeled outputs**, the algorithm discovers **inherent patterns** within the dataset.

### K-Means Clustering
K-Means is a method used to **group similar data points together into K clusters**.

![Day 17 kmeans](/images/Day_17_k_means_clustering.png)

1. **_Picking K random points as starting centers (centroids)_**.
2. **_Assigning each data point to the nearest centroid_**.
3. **_Updating the centroids by taking the average position of the points in each cluster._**
4. **_Repeating the process until the clusters stop changing._**

### Application of Clustering
- Grouping similar news
- DNA analysis
- Astronomical data analysis

___
## Day 18: Cost Function Optimization, Elbow Method

### Cost Function Optimization
In **unsupervised learning**, particularly in **K-means clustering**, the optimization process revolves around **_minimizing a cost function (distortion)_** that measure how well the clusters represent the data. The standard cost function for K-Means is the _**sum of squares (SSD)**_ between each data point and its assigned cluster centers. The cost function also called **inertia or with-cluster sum of squares (WCSS)** is defined in below images:

![Day 18 optimization](/images/Day_18_optimization_distortion.png)

- The objective of K-Means is to ***minimize this function by iteratively updating cluster assignments and centroids***.

### Elbow Method
The **Elbow Method** is a simple way to find the best **_number of clusters_** in K-Means. It works by measuring how much the data points in each cluster differ from their assigned center (called a centroid). The idea is:

- If you use **too few clusters**, the points within a cluster will be very far apart, meaning the groups aren’t well-defined.
- If you use **too many clusters**, each cluster will only have a few points, which isn’t useful.

To find the best number of clusters (k), we calculate a cost function called **WCSS** (**Within-Cluster Sum of Squares**).

![Day 18 elbow method](/images/Day_18_elbow_method.png)

In this above figure, in K = 3, the curve not bending too much. So, we have three cluster formation.

___
## Day 19: K-Means Clustering From Scratch in Python
KMeans Clustering can be used with open-source library scikit-learn. But to know about the working from depth, we should gain scratch implementation of this algorithm. Although, I had used scikit-learn to create a clustering dataset. 

![Day 19 make_blobs to create dataset](/images/Day_19_kmean_scratch_implement.png)

In this code, I had created a 100 datapoints using `make_blobs` function. Here, number of clusters are two and also iterate for 100 times to come up with accurate cluster centroids. I design a class named as **KMeans**. I initialize the constructor with `n_centroids` and `max_iter`. And assign `centroids` with `None` initially. Inside a **KMeans** class, I had define three functions such as `fit_predict`, `assign_cluster` and `move_cluster`. `fit_predict` is our main function where all training actions comes apart. And remaining two are the helping function.
- **assign_cluster:** In this section, I had find the **euclidean distances** for each datapoints from each cluster centroids. And find the minimum distances and assign its **index position** to the list named as `cluster_group`. `cluster_group` list consist of 100 elements. And lastly that function returns list of `cluster_group`.<br></br>
- **move_cluster:** The function is defined with all datapoints and list of `cluster_group` which we had got from the `assign_cluster` function. I separated all records with respect to its centroids.Then, calculate mean and assign with new centroids and return that value.<br></br>
- If you find the **old centroids** and **current centroids** remain same, then you can stop the iteration. Hence, with these brief ideas we can complete our scratch implementation of KMeans.
Below in images, you can look how I came with idea.
![Day 19 kmeans algo](/images/Day_19_scratch_implementation.png)

Output is shown below:

![Day 19 final output](/images/Day_final_output_kmeans.png)

*Resources*: [CampusX](https://www.youtube.com/watch?v=MFraC1JObUo) - ***KMeans Clustering Algorithm From Scratch in Python***

___
## Day 20: Anomaly Detection
**Anomaly Detection** is an **unsupervised learning** because anomalies are rare and labeled data is often unavailable. However, anomaly detection can also be approached using **supervised** and **semi-supervised** learning depending on the availability of labeled data.

In other word, **Anomaly detection** refers to the process of identifying data points, events, or patterns that significantly deviate from the normal or expected behavior in a dataset. These anomalies, also known as **_outliers_**, can indicate **_critical insights_**, such as **_fraud_**, **_system failures_**, **_cyber intrusions_**, or **_rare events_**. Some of the examples are shown in below images.

![Day 20 anamoly detection](/images/Day_20_anamoly_detection.png)

### Density Estimation
Before diving deeper into **anomaly detection** we should first understand about the **density estimation**. Studying **density estimation** first helps build intuition about what is **_normal_**, how **_anomalies stand out_**, and how **_models can identify them_**.

![Day 20 density estimation](/images/Day_20_density_estimation.png)

- **Density estimation** is the process of determining a probability distribution from a given dataset. It aims to estimate the **probability density function (PDF)** of a **_continuous random variable_** without assuming a predefined distribution. So, let's discuss a well known normal distribution below which a **parametric density estimation**.

### Normal (Gaussian) Distribution
![Day 20 normal distribution](/images/Day_20_normal_distribution.png)

**Normal distribution** is one of the most important probability distributions we'll learn about since a countless number of statistical methods rely on it. It applies to more real-world situations than  other distributions. Its shape is like **bell curve** as shown in above picture. 

**Important properties**
- It's symmetrical so left side is a mirror image of the right.
- **_The area beneath the curve is 1_**. `(Area =1)`
- The probability never hits `0`, even if it looks like it does at the **_tail ends_**.
- Describe by **_mean_** and **_standard deviation_**.

When a normal distribution has **_mean_** `0` and **_standard deviation_** of `1`, it's a special distribution called the _**standard normal deviation**_. As it's illustrated into below image.

![Day 20 variation with mean and std](/images/Day_20_variation_with_mean_std.png)

In above, if mean changes the position then curve will shift and increases in standard deviation makes the curve flat. And **squeezes of curve** when the **_standard deviation_** tends to decrease. With these **statistical ideas** we are good to go with the anomaly detection algorithm. Tomorrow, I'll explore the **algorithm of anomaly detection**.

___
## Day 21: Anomaly Detection Algorithm
**Gaussian distribution** is the crucial part for this algorithm. There is a value called **_epsilon_**. If pdf of normal distribution is less than **_epsilon_** then algorithm detected anomaly and if pdf is greater than epsilon then the detected point is **normal** (**non-anomalous**). First of all, we need to take all the normal datapoints while training model because it contains skewed data. However, there could be a misconception between **anomaly detection** and **supervised learning**. So, if our dataset split into the ratio of 99:1 then that problem would be anomalous detection. If dataset split into nearly equal proportion then obviously we should choose supervised classification model. Let's look out into more mathematical and stepwise algo in image below:

![Day 21 anomaly algorithm](/images/Day_21_anomaly_algorithm.png)

Taking two features and utilizing algorithm, we can clearly see into the image below:
- ![Day 21 example based on anomaly](/images/Day_21_anomaly_detection_example.png)


- ![Day 21 aircraft example](/images/Day_21_anomaly_aircraft_example.png)

However, in real-world it is impossible to find well known distribution such as **_Gaussian distribution_**. But we can achieve that known distribution with feature transformation.
![Day 21 feature transformation](/images/Day_21_features_transformation.png)

___
## Day 22: Collaborative Filtering
Yesterday, I had completed _week 1_ from the last part of **Machine Learning Specialization**. And today, I explored the recommended system. Learn about the collaborative filtering, regularization term and gradient descent in recommender systems.

![Day 22 collaborative filtering](/images/Day_22_colaborative_filtering_mathematics.png)

**Collaborative Filtering** is a technique used in **recommendation systems** that predicts user preferences by analyzing **_past interactions_** and **_similarities between users_** or items. It assumes that similar users or items will have similar preferences.

Below images show about the movie recommendation system. This image show the scratch implementation. Although, mathematical formulae coincides with linear regression.
![Day 22 recommendation system](/images/Day_22_collaborative.png)

![Day 22 collaborative notation](/images/Day_22_collaborative_notation.png)
Above shows the notation used for movie recommendation in image.

![Day 22 cost function](/images/Day_22_cost_function_of_recommender.png)
Included the regularization term same as linear ridge regression.

![Day 22 gradient descent](/images/Day_22_collaborate_gradient_descent.png)

___
## Day 23: Mean Normalization, Tensorflow Implementation of Recommendation System
In recommendation systems, **Collaborative Filtering** works by predicting user preferences based on the preferences of similar users. However, different users have different rating scales. **Mean normalization** helps **_remove these biases and improve similarity calculations_**.
![Day 23 mean normalization](images/Day_23_mean_normalization.png)

**Why Use Mean Normalization?**
- Different users rate items on different scales.
- Some users may give high ratings to all movies, while others rate more conservatively.
- Mean normalization removes user-specific biases, making it easier to compare users and items.
- Helps compute similarity metrics (cosine similarity, Pearson correlation) more accurately.
- Essential for Netflix, Amazon, and YouTube recommendation systems.

### TensorFlow Implementation of Collaborative Filtering

![Day 23 implementation](/images/Day_23_tensorflow_implementation.png)

**Collaborative filtering** is a key technique in recommendation systems, aiming to predict missing user-item interactions based on observed data. Andrew Ng, in the **Machine Learning Specialization**, explains its implementation using **matrix factorization with gradient descent** in TensorFlow. This approach represents users and items as **low-dimensional latent vectors**, where the dot product of these vectors approximates user ratings. The model optimizes embeddings by minimizing the **Mean Squared Error (MSE)** loss, focusing only on known ratings to enhance learning accuracy. Training is conducted using advanced optimizers like **Adam**, iteratively refining embeddings for better predictions. This method is widely applied in industry, including Netflix and Amazon, to personalize user experiences. While matrix factorization is effective, deep-learning-based models such as **Neural Collaborative Filtering (NCF)** offer further improvements by capturing complex user-item relationships.

![Day 23 custom training loop](/images/Day_23_custom_training_loop.png)
### Finding Related Items
In recommendation systems, finding related items is an essential task for providing personalized suggestions to users. This typically involves identifying items that are similar to a given item, based on the preferences of users or the characteristics of the items themselves.

Finding related items in **recommendation systems** is a combination of calculating item **_similarity using either item features (content-based) or user interactions (collaborative filtering)_**. Advanced methods like matrix factorization provide a powerful way to uncover latent patterns and improve the relevance of related item recommendations.

![Day 23 finding related item](/images/Day_23_finding_related_item.png)

___
## Day 24: Content-Based Filtering
Today, I dived into another approach for recommender system. Let's explore about two distinct approach:

**Collaborative Filtering**
* This approach recommends items based on the **_interactions and preferences_** of other users with similar tastes.
* It assumes that if two users have **_similar past behavior_**, they will **_likely prefer similar items_** in the future.
* It is divided into:
  * **User-based Collaborative Filtering**: Finds similar users and recommends items liked by them.
  * **Item-based Collaborative Filtering**: Finds similar items based on user interactions and suggests items similar to what the user has interacted with.
* Example: If User A and User B both like Movie X, and User B also likes Movie Y, then User A is recommended Movie Y.

**Content-Based Filtering**

![Day 24 content based filtering](/images/Day_24_content_base_filtering.png)
* This approach recommends items based on the **_characteristics of the items and the preferences_** of the user.
* It relies on item features (e.g., genre, keywords, descriptions) and user profiles.
* It uses techniques such as **_TF-IDF (Term Frequency-Inverse Document Frequency)_** and **_cosine similarity_** to measure how similar items are to what a user has liked before.
* Example: If you watch many action movies, the system recommends other action movies with similar features.

Yesterday, we had completely learnt about the collaborative filtering. Today, I had explored content based filtering. Lets deep dive:

The content based filtering consist of two crucial steps:
They are Retrival and Ranking:

**Step 1: Retrival**
![Day 24 retrival and ranking](/images/Day_24_retrival_step_1.png)
The retrieval step focuses on **_selecting a subset of relevant items from a large pool_** of available content. Since searching through an entire catalog can be _**computationally expensive**_, this step efficiently filters down the dataset to a manageable size.

**Step 2: Ranking**
![Day 24 ranking step](/images/Day_24_ranking_step_2.png)
Once the retrieval step provides candidate items, the ranking step **_sorts these items in order of relevance to the user_**. A ranking algorithm determines which items are most relevant to the user’s interests.

**Ethical Use of Recommender System:**
![Day 24 ethical use](/images/Day_24_ethical_use_recommender_system.png)

**Some goals of recommender system:**
![Day 24 goals of recommender](/images/Day__24_goal_of_recommender_system.png)
These are the application of recommender system. However, it helps in profit making but should maintain ethnicity.

At last, I learnt about the implementation of **content based filtering** approach using **_Tensorflow framework_**.

![DAy 23 implementation](/images/Day_24_content_bas_filtering_implement.png)
Will further discuss on implementation in later coming days.

___
## Day 25: Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a **_dimensionality reduction technique_** used in data science and machine learning. It transforms a **_high-dimensional dataset_** into a **_lower-dimensional space_** while preserving as much variance (information) as possible.

![Day 25 pca algorithm](/images/Day_25_PCA_algorithm.png)

If we draw the scatter plot between data and using trial and error concept to ensure axes which can **_capture quite a lot of the spread of the data_**. Basically, **PCA** helps to find the **axes** which provides a **_larger variance_** capturing more information of original data with **_few dimensions_**.

Let's look into the below images which show the scikit-learn implementation of PCA algorithm:

![Day 25 scikit learn pca algo](/images/Day_25_pca_in_sklearn.png)

**Code of PCA in scikit-learn:**
![Day 25 code of PCA](/images/Day_25_code_sklearn.png)

In above, the 2-dimensional dataset is converted into 1-dimensional data which **_describe 0.992 ratio of variance of original data_**.

___
## Day 26: Visualize Decision Tree
Today, I had revised the decision tree concept and also explored about decision tree visualization using scikit-learn library. `plot_tree` function is used to visualize the tree diagram which gives us a better intuition. I also learned to customize tree diagram using some of the attributes. 

![Day 26 decision tree plot](/images/Day_26_plot_decision_tree.png)

Also explored the one of the important visualization technique i.e. ***pie chart*** using matplotlib library. Here, attributes such as `startangle`, `explode `, `autopct`, `label` and `color` are used for customizing pie chart. For plotting the pie chart, I used builtin dataset named as iris-dataset. 

![Day 26 pie chart](/images/Day_26_pie_chart_iris.png)

I had completed the second last week from the third part of **Machine Learning Specialization**. I will dive deeper to last week i.e. *week 3* from tomorrow onwards. In last part, I will learn about the basic concept related to reinforcement learning.

Today's Notebook: [Practice decision tree session](Day_26_decision_tree_practice)
___
## Day 27: Introduction of Reinforcement Learning

**What I learnt:** I explored introduction of reinforcement learning and learned some of its terminologies. Also, gain insights on MDP stands for **Markov Decision Process**. All things are describe in detail below:

### Introduction

![Day 25 reinforcement learning](/images/Day_27_reinforcement_learning.png)
**Reinforcement Learning (RL)** is a type of machine learning where an **_agent learns_** to make decisions by interacting with an **_environment to achieve a goal_**. The agent takes **_actions, receives feedback_** in the form of **rewards**, and updates its strategy to _**maximize cumulative rewards**_ over time.

### Some of the terminologies
* **Agent** – The learner or decision-maker.
* **Environment** – The system the agent interacts with.
* **State (s)** – The current situation of the agent.
* **Action (a)** – The choices the agent can make.
* **Reward (r)** – A numerical signal given as feedback.
* **Policy (π)** – A strategy that maps states to actions.

**Return** - It refers to the **total accumulated reward** an agent receives from a given state until the end of an episode. It is used to evaluate how good a sequence of actions is.

![Day 27 return on RL](/images/Day_27_return_on_RL.png)

### Markov Decision Process:
![Day 27 mdp](/images/Day_27_markov_decision_process.png)

A **Markov Decision Process** (MDP) is a **mathematical framework** used to model decision-making problems where outcomes are partly random and partly under the control of an agent. MDPs are widely used in **Reinforcement Learning (RL)** to model the interaction between an agent and an environment.

___
## Day 28: State Action Value Function
The state value function, denoted as `V(s)`, _**measures how good it is for an agent to be in a particular state**_ `s` while following a specific policy `π`.

Formally, the state value function is the **expected total reward** an agent can obtain starting from state `s` and following policy `π` thereafter.

![Day 28 state action value function](/images/Day_28_state_action_value_function.png)

- `Q(s, a)` = Return if you
     - start in state s
     - take action `a` (once)
     - then behave optimally after that

where, <br>
`s`: current state<br>
`a`: current action<br>
`R(s)`: reward of current state<br>
`s'`: state you get to after taking action a.<br>
`a'`: action that you take in state `s'`.

### Bellman Equation
![Day 28 bellman equation](/images/Day_28_bellman_equation.png)
The **state-action value function** already tells us how good an action is in a given state, so why Bellman equation needed.

_Here's some reasons:_
* Instead of estimating values from scratch, the Bellman equation breaks the problem into smaller parts.
* It tells us that the value of a state/action depends on immediate rewards + future values.
* The Bellman equation helps update values over time as the agent explores new states and so on.

At terminal state, `Q(s, a)` = `R(s)`


**Conclusion**
- The Bellman equation is a _**recursive formula that helps in decision-making**_.
- It breaks down a problem into smaller sub-problems, making it easier for RL agents to learn.
- It forms the basis of many RL algorithms like **Q-learning and Deep RL**.

___
## Day 29: Continuous State Space, Deep Reinforcement Learning

Previously, I explored on single state, one state at a time for finding current state action value function. Also, explored on discrete state which only takes **quantized value**. The state could be +ve integer number. Today, I dived into the continuous state which contains position, speeds and twist of particular object. They are **written collectively in the form of vector** which contain different attributes ase mention in above.

I understand continuous state using some of the real world based project such as **autonomous helicopter**, **lunar lander**. Mostly, **Andrew Ng** explain this section with the help of lunar lander project.

For lunar lander, state is a vector which contains 8 dimension vector such as x-position, y-position, change in x-position, change in y-position, twist/rotation and change in rotation. And lastly, contain two boolean element of vector named as l and r. This attributes l and r show the lunar lander left legs and right legs touches or not on the surface respectively. This lunar lander contain actions such as nothing, left, main, right based on producing trust in particular direction.

![Day 29 learning algorithm](/images/Day_29_learning_algorithm_reinforce.png)

### Deep Reinforcement Learning
This reinforcement learning can be implemented with the help of neural network. However, it is not optimized and can enhance optimization using output layers contain 4 nodes instead of using 1. For that, input doesn't contain actions.

![Day 29 reinforcement learning](/images/Day_29_reinforcement_deep_learning.png)

![Day 29 q-learning](/images/Day_29_q-learning.png)
- Algorithm can be refined using **mini-batch gradient** and **soft updates**.

### Limitation of Reinforcement Learning
1. Much easier to ge to work in a simulation than a real robot.
2. Far fewer applications than supervised and unsupervised learning.
3. But ... **_exciting research direction with potential for future applications_**.

Finally, learning all these stuff comes to an end. I had completed **Machine Learning Specialization** after learning some of the ideas behind the reinforcement learning in last week of this session. However, I will continue on working some of the projects and revised all this stuffs for the better understanding. 

___





