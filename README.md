[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Build an RL agent that collects Bananas

### Introduction

For this project, I will train an agent to navigate (and collect bananas!) in a large, square world. Unity Machine Learning Agents (ML-Agents) plugin will be used to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents)



![Trained Agent][image1]

### Environment & Goal 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

#### Install packages and dependencies

For the Training, I used a p2.xlarge type AWS EC2 instance (Ubuntu based Deep Learning AMI, AMI ID i-0aeca4a4e5d610469) the Seoul Region, where I closely located.  Most of the utilities are already installed  in Deep learning AMI so minor correction was made on requirment.txt file. 



1. Create and activate a new conda environment 

   ###### Linux or Mac:		

   ```
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

   ###### Windows:

   ```
   conda create --name drlnd python=3.6
   activate drlnd
   ```

2. Install OpenAI gym in the environment

   ```
   pip install gym
   ```

3. Install extra environment groups. 

   ```
   pip install 'gym[classic_control]'
   pip install 'gym[box2d]'
   ```

4.  Clone following repository and install additional dependencies.

   ```
   git clone https://github.com/jihys/udacity-rl-project1.git
   #Alterntively you can download original codes from udcity github. 
   #git clone https://github.com/udacity/deep-reinforcement-learning.git`
   cd python`
   vi requirement.txt` 
   "Correct requirement.txt as needed"`
   pip install .
   ```

   Note: Please comment out jupyter,ipykernel in the "deep-reinforcement-learning/python/requirment.txt" file and install required packages. 

   ```
   tensorflow==1.7.1
   Pillow>=4.2.1
   matplotlib
   numpy>=1.11.0
   #jupyter
   pytest>=3.2.2
   docopt
   pyyaml
   protobuf==3.5.2
   grpcio==1.11.0
   torch==0.4.0
   pandas
   scipy
   #ipykernel
   ```

#### Unity Environment Setup 

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the `udacity-rl-project1/` folder, and unzip (or decompress) the file. 

    

### Instructions

Follow the instructions in `Project1_Navigation.ipynb` to get started with training your own agent!  There you will find how to call agent with different DQN algorithms. I used DQN, Dueling DQN, Double DQN for this project. 

`model.py/` defines neural network that estimates Q values. Change the number of hidden layers and nodes as you wish.

`dqn_agent.py` includes Agent class and Replay Buffer class that is used to interact and train agent. This code contains code that does followings:

* Initialize agent and replay buffer

		* Train agent with simulation env. ( Here we will use Unity ML agent.)
		* Define Q value update logic 

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `udacity-rl-project1/` folder and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.