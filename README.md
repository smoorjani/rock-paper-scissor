A computer vision based game of rock, paper, scissors with a CNN.

# Inspiration

Just as many beginners build off of a simple program and make it more interesting, we wanted to take something simple and transform it into something much more fun. For this reason, we created Rock, Paper, Scissors, except you play with your computer in a much more convenient way. 

# What It Does

Lonely Rock Paper Scissors (LRPS) allows you to play a game of Rock, Paper, Scissors with your computer without having to do anything except play. There is no entering your move or telling the computer when to go. It all happens automatically.

# How We Built It

Our first step was to find a way to get imaging data from a webcam. We used OpenCV to capture webcam feed and then compare it to a constant background in order to segment the player's hand.

We then had to classify the hand as either rock, paper, or scissors. We did this through a convolutional neural network, trained on data we made using our recognition program and tested on separate data. We accomplished a 92% accuracy rate for classifications.

Lastly, we put it all together, using the data we collected from the capture during a session and classifying it through the CNN in order to play the game.

# Challenges

For the most part, the challenges came from our environment. With only 12 hours, it was hard to build a comprehensive model and even harder to make it work with greater detail. Our image data was formatted into a size of 64px by 64px, and the CNN had sparse hidden layers with only 32 nodes in each while our input had 4096 nodes.

# What's Next

We plan to expand the game to Rock, Paper, Scissors, Lizard, Spock (the one from Big Bang Theory). One of the challenges we plan to face is distinguishing Rock from Lizard which may require more detailed imaging. Although what we created is just for entertainment purposes, this proof-of-concept project can be used for so much more whether it's diagnosing medical conditions or evaluating crop health.
