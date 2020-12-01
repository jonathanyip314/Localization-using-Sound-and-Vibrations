# Localization-using-Sound-and-Vibrations

This python script is used to localize events based on sound and vibrations. 
Instructions to use this code.
1. Create a directory containing data from 3 microphone sensors and 2 imu sensors
  Be sure to have the sensor data files in the order: 01,02,03,04,and 05
2. on line 484, point the dir variable to the folder
3. In lines 408 and line 421 adust the number of events detected by modifying the Events array indicies. Remove indicies to detect and localize all events.
4. If the files contain events that all occur at one location you can adjust the ground truth location at line 446
  this will give an error value indicating how far off our estimations are to the ground truth
5. Run the code.
  A graph will pop up that shows all the data
  the console should indicate each print and how to continue
