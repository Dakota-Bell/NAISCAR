# NAISCAR
An AI project to predict who will win a race in the 2025 NASCAR season

- Get user input for:
    * Specific Driver
    * What track the user wants information about
    * Estimated starting position (after qualifying you can input the real starting position)

- The output:
    * Determining whether the driver will win (Yes, Maybe, No)
    * Predicted finishing position of the chosen driver at that track

- The way the program works (currently) is by taking information from the previous two seasons (2023-24)
  and using a Random Forest Regressor (RGR) by taking in all the data from each season and placing it
  into one single data-point.
- With the data from the that variable it can then be split into training/testing variables to be placed
  into the RGR. After the model is trained it can be used to make a prediction using the Mean Squared 
  Error and the R-squared value to test the accuracy of the model.
- The script will ask the user for inputs and make a prediction based on the driver, track, and starting
  position for the year 2025 based on the previous seasons data. The script will then display the output
  based on this information

- Todo:
    * Gather data for average lap time
<<<<<<< HEAD
    * Gather data for average pit stop time
    * Implement information about the NASCAR playoff system. 
    * Implement live-stats (probably way too optimistic for my current skill-set)
=======
    * Gather data for average pit stop time
    * Implement information about the NASCAR playoff system. 
    * Implement live-stats (probably way too optimistic for my current skill-set)
>>>>>>> 05aab3026670fde011fc806e1a8eaad75ae78ff0

