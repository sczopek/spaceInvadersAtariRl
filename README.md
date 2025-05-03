# Space Invaders Project Summary

I trained an AI Agent to play Atari’s Space Invaders using PPO (Proximal Policy Optimization).  (PPO is a Reinforcement Learning algorithm that learns through self play.)  

The agent plays Space Invaders better than an average skilled human.  The agent can get the game score to roll over twice.  (The score rolls over when it exceeds 9999.  This score rollover occurs every 16 levels, so the agent has completed the first 32 levels of the game when the score rolls over twice.  Space Invaders has no final level like many other Atari games, so it is possible to play the game indefinitely.  In practice my agent can reach as far as level 32 in its highest scoring game.)

My agent is self taught and plays Space Invaders using only the images shown on the screen.  The agent has no prior knowledge of the game and has learned to play through self play.  This is similar to what [Google’s Deepmind team did with Atari’s Breakout game](https://youtu.be/V1eYniJ0Rnk?si=MxJzxsX09T2sNEiW).

I am proud of what I accomplished and wanted to share my results.


## Results 

**High Score:** 24.9 k  (24,935)  

**Mean Score:** 5.0 k

**Median Score:** 4.3 k

**Difference between Mean and Median:** 0.22 std dev, ~1 game level

**Sample Size:** 500 Games 

**Std Dev:**   3.4 k

**Percentage Games Where Score Rolls Over:** 8.4%

**Percentage Games Where Score Rolls Over Twice:** 0.6%


 
Video of High Scoring Game:

<img src="space_invaders_ppo__score_24935.gif" width="700" height="700"/>


## Agent Model 

This is a summary of how my agent chooses its Atari controller input for each frame of the game.
- Prprocessing Layers
- Action Selection with CNN (Convolutional Neural Network)
- PPO Training Philosophy

## Preprocessing Layers  
- Casting Images to Black and White
- Rescaling to Square Images
- Frame Stacking

### Casting Images to Black and White
The raw game images are taken as model input from the ALE (Atari Learning Environment) emulator.  The raw images are then cast from color images into black and white images.  The rational for the loss of color is that black and white images are faster to train on.  A black and white image can be respresented by a single number, but a colored RGB image is represented by three numbers.  Additionally, the color images do not carry any additional information so using black and white does not result in a loss in information.  

### Rescaling to Square Images
The images are then made square after being made black and white.  These square images also reduce model training time, because the agent does not need to learn that the images are rectangular when fitting certain multi-dimensional gaussian functions*.  

### Frame Stacking
Frame stacking is used to help the agent detect motion on the game screen.  A raw screenshot does not indicate velocity for objects in the image frame.  A short image history buffer is used to infer object motion.   This motion detection behavior is automatically learned by the next CNN Layer.  

The specifics of my framestacking scheme is to store and stack the last four images as a tensor.  These four images are the current image and three previous image.  Each image has a four frame spacing between images, images (N - 12, N - 8, N - 4, N).  This is the output that is piped into the next CNN layer.


## Action Selection with CNN 
### How the CNN Selects an Atari Controller Command 

The CNN layers accepts a stack of images as input and then performs self-taught feature extraction.  This feature extraction is performed by sliding a window across the stack of images and performing a convolution of the sub-image in that sliding window.  This convolved product, of the sub-image, forms the next image layer, which is then convolved again with a separate sliding window that belongs to the next convolutional layer.  This process is repeated several times.  

To achieve the final controller output, the final convolutional layer is flattened and passed through a dense, fully connected layer that maps the different extracted layers to Atari controller input probabilities.  (This is done with matrix multiplication of a weight matrix against the flattened feature vector CNN output.)  Some details are omitted for brevity.  The agent then chooses an action based on which probability has the highest chance of increasing the total score for the Space Invaders game.

### CNN Network Description
The CNN used is from the original nature paper.
- 84x84 Greyscale (single channel) Image
- Framestacking
- 1st CNN, 8x8 Window, 16 Channels Out, Stride 4, No Padding
- Relu
- 2nd CNN, 4x4 Window, 32 Channels Out, Stride 2, No Padding
- Relu
- 3rd CNN, 3x3 Window, 64 Channels Out, Stride 1, No Padding
- Relu
- Flatten
- Fully Connected Layer, 256 x Number of Actions
- Relu 

### Calculating the Receptive Field Size

The Receptive Field (RF) size indicates the width of the largest feature the CNN can detect.  The Receptive Field depends on the kernel dimensions for each CNN layer.  The larger the kernel width, the larger Receptive Field.  This is also true for the number of CNN layers too.  The more CNN layers there are, the larger the recpetive field.  (MaxPooling, a type of downsampling, also influences the Receptive Field size.  Kernel Stride also influences RF size too.)


RF Size = 36 Pixels
Game Screen = 84x84 Pixels
RF Size / Screen Width = 36/84 = 42.9%

The RF size does not have to span the entire screen width (84 pixels) to select a good next action for the Space Invaders game.
The RF Size can contain several Space Invaders game sprites, so the CNN output features can simultaneously consider several objects at once.

RF Formula:
j_out = j_in * s
r_out = r_in + (k-1) * j_in

These values are calculated recursively for each CNN Layer.

r = receptive field
k = kernel width, if kernel is 3x3 k=3
s = kernel stride
j = kernel jump.  Kernel jump for layer n only depends on the previous (n-1) layers.

RF Size Values for Each Layer:
-RF, Input = 1
-RF, Layer 1 = 8
-RF, Layer 2 = 20
-RF, Layer 3 = 36


## PPO Agent Training

[Please see this Open AI's announcement for details.](https://openai.com/index/openai-baselines-ppo/)

[Or read this thesis for even more technical details.](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf)

PPO (Proximal Policy Optimization) is a training technique.  PPO's goal is to explore the environment to find a good Policy.  A Policy is a algorithm that looks at the screen and then provides the odds about what choice is best.  The agent chooses between different gameplay actions, and for a given scene action 1 is best (on avg) some percentage of the time, action 2 is best a different percentage of the time, etc.  

The agent learns to predict this policy by exploring the environment.  This is done by playing through the space invaders game many times.  For each screen capture (each frame) the agent predicts which action is best and then compares that prediction to the observation.  The difference between prediction and observation is the basis for the agent's policy update.

PPO's innovations over earlier RL techniques is twofold.  It only allows small (proximal) policy updates.  The PPO algorithm uses a much simpler small policy update than its immediate predecessor TRPO (Trust Region Policy Update).  It is not clear to me why these simple innovations produce better results for PPO, but there is emprical evidence that shows PPO is a robust RL learner that can demonstrate model improvement in situations where other RL techniques fail.

## Python Modules Used for Project

Gymnasium: Provides API interface between the ALE Atari games and the RL package.
https://gymnasium.farama.org/

ALE (Atari Learning Environment): Provides the Atari Space Invades game with an API to assist with ML (machine learning).
https://ale.farama.org/

StableBaselines-3: Contains a library of reinforcement learning agents that can easily be connected to cloud compute resources allowing for easy training an gameplay.
https://stable-baselines3.readthedocs.io/en/master/index.html


## Project Costing

### Cloud Compute Costs
$49.99/month, Google Colab Premium
Project time 5 months.
~$250 total

This total $250 project cost is at most a couple of engineering man hours, suggesting that training costs are not the limiting cost factor for training task with small datasets. 


# Additional Anaysis

## Agent Performance Plots

A blue dashed line indicates where game score roll over occurs.  A game's score rolls over when it exceeds 9999.

### AI Agent Score PDF
![](./readmeImgs/sortedRewards_gameScore_vs_gameCount.png)

![](./readmeImgs/scoringPdf_wSmoothing.png)

The orange curve is the smoothed PDF function.  The raw score jumps in increments of 5, 10, 15, 20, 25 and 30, which leads to a jagged PDF with many discrete gaps.

### AI Agent Score CDF
![](readmeImgs/cdfScore.png)


