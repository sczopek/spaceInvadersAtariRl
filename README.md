# Space Invaders Project Summary

I trained an AI Agent to play Atari’s Space Invaders.  The agent can get the score to roll over twice.  (The score rolls over when the score exceeds 9999.  This score rollover occurs around every 16 levels, so when the score rolls over twice the agent has completed the first 32 levels of the game.  Like many Atari games Space Invaders has no final level.  It is possible to play indefinitely.)

My agent is a generalized agent.  It is self taught and plays Space Invaders using only the images shown on the screen.  The agent has no prior knowledge of the game and learned through self play.  This is similar to what [Google’s Deepmind team did with Atari’s Breakout game](https://youtu.be/V1eYniJ0Rnk?si=MxJzxsX09T2sNEiW).

I am proud of what I accomplished and wanted to share my results.


## Results 

**High Score:** 24.9 k  (24,935)  

**Mean Score:** 5.0 k

**Median Score:** 4.3 k

**Std Dev Difference:** 0.22 std dev between mean and median scores

**Sample Size:** 500 Games Played 

**Std Dev:**   3.4 k

**Percentage Games Where Score Rolls Over:** 8.4% of games

**Percentage Games Where Score Rolls Over Twice:** 0.6%


Video of High Scoring Game:

<img src="space_invaders_ppo__score_24935.gif" width="700" height="700"/>


## Agent Model 

This is a summary of how my agent chooses its Atari controller input for each frame of the game.

The preprocessing layers.  
•	Casting to Black and White
•	Rescaling to Square Image
•	Frame Stacking

### Casting Images to Black and White
The raw game images are taken from the ALE (Atari Learning Environment) emulator, and then cast from color images into black and white.  Since, the color images do not carry any information using black and white inputs allow the next layer, the CNN, train faster.  

### Rescaling to Square Images
After casting the raw images into black and white the images are then made square as these square images also reduce model training time.  The agent does not need to learn that the images are rectangular when fitting certain multi-dimensional gaussian function under the hood*.  

### Frame Stacking
Frame stacking is used to store a short image history which helps the agent detect motion on the screen.  A raw screenshot does not indicate velocity for objects in frame.  When the current image and three previous images are stored in a buffer motion can be inferred.   This motion detection behavior is automatically learned by the next CNN Layer.  
The specifics of my framestacking scheme is to store and stack the last four images as a tensor.  Each image is four frames before or four frames after the next image.  This is the output that is piped into the next CNN layer.


## CNN (Convolution Neural Net)

The CNN layers accept a stack of images as input and then perform self-taught feature extraction.  This feature extraction is performed by sliding a convolution window across the stack of images.  This convolved product forms the next image layer, which is convolved is convolved again with a separate sliding window that belongs to the next convolutional layer.  This process is repeated several times.  

To achieve the final controller output, the final convolutional layer is flattened and passed through a dense, fully connected layer that maps the different extracted layers to Atari controller input probabilities.  (This is done with matrix multiplication of a weight matrix against the flattened feature vector CNN output.  Some details are omitted.)  The agent then chooses an action based on which probability has the highest chance of increasing the total score for the Space Invaders game.

## PPO Agent Training

[Please see this Open AI announcement for details.](https://openai.com/index/openai-baselines-ppo/)

[Or read this thesis for even more technical details.](https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf)

## Python Modules Used for Project

Gymnasium: Provides API interface between the ALE Atari games and the RL package.
https://gymnasium.farama.org/

ALE (Atari Learning Environment): Provides the Atari Space Invades game with an API to assist with ML (machine learning).
https://ale.farama.org/

StableBaselines-3: Contains a library of reinforcement learning agents that can easily be connected to cloud compute resources allowing for easy training an gameplay.
https://stable-baselines3.readthedocs.io/en/master/index.html


### Cost Project

## Cloud Compute Costs

## Engineering Man Hours

