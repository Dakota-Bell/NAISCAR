# NAISCAR
An AI project to predict who will win a race in the 2025 NASCAR season

* Get user input for:
    1. Specific Driver
    2. What track the user wants information about
    3. Estimated starting position (after qualifying you can input the real starting position)

* The output:
    1. Determining whether the driver will win (Yes, Maybe, No)
    2. Predicted finishing position of the chosen driver at that track

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
    1. Gather data for average lap time
    2. Gather data for average pit stop time
    3. Implement information about the NASCAR playoff system. 
    4. Implement live-stats (probably way too optimistic for my current skill-set)

