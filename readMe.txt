Snake Deep-Q Convolutional Nerual Network

===================================================
The main.py script initiates the training process:

1. Checks to see if there is a graphics card avalible to use, otherwise uses CPU (and prints the device it will be using for training)

2. Initiates the snake environment wrapper in the wrapper.py
    this is a preprocessing step for information from game going to our agent. 
        for different games you can set a repeat (not necessary for snake)
        Immage preprocessing and reshaping happens here
        The reset() function for the game is here
        This script communicates with the game.py file which is where the actual game lives and that is where the reward functions are defined

3. The AtariNet model class is initiated with 4 actions for snake (up, left, down, right) in model.py
    a base network or dueling network is created based on perameters set in settings
    This is where the save and load functions live

4. The agent class is initiated with training perameters
    the replay memory is initiated here (that also lives in agent.py)
    the agent can communicate between the network and the environment with the get_action() function
    a plotter class is initiated to keep track of progress during training (plotting class in plot.py)
    training loop with various checkpoints to output information through training (usually 200k epochs)


===================================================
The test.py script initiates the testing process:

mostly the same set of steps as with training, but certain perameters (such as randomness in decision making) are switched for model evaluation


===================================================
Best practices for running this code are as follows

1. install requirements.txt

2. adjust perameters in the settings.py file

3. create a note in the notes folder with the settings info and the nohup command you used to run this process in the background (so that training is easily repeatable)

4. console output is sent to the runs directory

5. plots are created and updated in the plots directory

there is also a playableSnake.py file for a human that wants to try their hand and beating the model in the game