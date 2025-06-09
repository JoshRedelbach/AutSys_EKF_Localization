# Extended Kalman Filter for Localization of a TurtleBot3 Waffle Pi

This repository includes all code and data used within the project _Extended Kalman Filter for Localization of a TurtleBot3 Waffle Pi_ for the course _Autonomous Systems_.
It was created by Carolina Realinho Pires, William Clarke and Joshua Redelbach.
The project report about the algorithm as well as the results can be read **PUT REFERENCE HERE THEN**

## Repo Structure

### Data
The [Data](/Data/) folder includes the used map of the laboratory environment as well as the recorded rosbags of the first two test trajectories. The last one is too large to be able to upload it here.

### catkin_ws
The folder [catkin_ws](/catkin_ws/) includes the code of the [EKF ROS node](catkin_ws/src/ekf/scripts/filter_node.py) as well as the code used by the [package](catkin_ws/src/ekf/ekf/) - all implemented in Python.
In order to execute it, follow the following steps:

_Prerequisite_: working Ubuntu 20.04 and ROS 1 version installed.
1. Download the folder.
2. Download the map files [full_room_01062025.pgm](Data/full_room_01062025.pgm) and [full_room_01062025.yaml](Data/full_room_01062025.yaml)
3. Adjust the path of the map files in [filter_node.py](catkin_ws/src/ekf/scripts/filter_node.py).
4. Install the following Python packages on your machine:
   1. numpy
   2. scipy
   3. opencv
   4. PIL
   5. yaml
5. Navigate in the terminal to the catkin_ws folder.
6. Run `catkin_make`
7. Run `source devel/setup.bash`
8. Run the node: `rosrun ekf filter_nody.py`

Additionally, in the file [filter_node.py](catkin_ws/src/ekf/ekf/EKFJ.py) it can be decided if the standard ICP or the ICP with the GO-ICP reinitialization routine is used. If the standard ICP is chosen, please select in the _icp\_pipeline(...)_-function (line 300) in [Benchmark_ICP.py](catkin_ws/src/ekf/ekf/Benchmark_ICP.py) which ICP algorithm shall be called.

### Results
The folder [Results](/Results/) includes the resulting .csv files of the simulated trajectories with .txt files that describe the used initial parameters of the filer. The sub-folder structure makes clear, which algorithm combination was executed to obtain this estimated trajectory.
Furthermore, it includes a folder [analysis](/Results/analysis/) that includes scripts to analyze the .csv files. Please pay attention and select the correct file paths.

### MicroSimulator
The folder [MicroSimulator](/MicroSimulator/) includes a microsimulation, which was developed in early stages of the project to validate the ICP algorithm on synthetic data. 