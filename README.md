# Autonomous Reinforcement Learning Based Drift Control for Abrupt Obstacle Avoidance Using Professional Racing Techniques
*(Submitted) IEEE Robotics and Automation Letters*


<a href="https://www.youtube.com/watch?v=VIDEO_ID"><img src="http://img.youtube.com/vi/VIDEO_ID/0.jpg" alt="Video Thumbnail" width="240" height="180" border="10" /></a>
## Requirements
1. Tested on Ubuntu 20.04.
2. Nvidia GPU equipped, and driver Installed. Tested on RTX 4080.
3. Install [Anaconda](https://www.anaconda.com/).
4. Install the environment:
```console
conda env create -f environment.yaml
```
This command will create a conda environment named `carla`



## Start the Simulator
We build the simulator based on [Carla 0.9.14](https://carla.readthedocs.io/en/0.9.14/getting_started/). 
Open a new terminal and start the simulator:
```console
cd PATH_OF_CARLA
./CarlaUE4.sh 
```

## Test the Model

To test the models, make sure you have started the simulator, then open a new terminal and do the followings:

```console
cd code
conda activate carla
python test.py
```

Then the model will be tested. The driving data (speed, location, heading, slip angle, control commands, etc.) will be recorded in `code/test/` after the testing process.


## Citation

Please consider to cite our paper if this work helps:
```
To be Added
```
