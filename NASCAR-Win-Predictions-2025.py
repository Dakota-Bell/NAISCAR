#Dakota Bell
#Description: AI program that asks user for the name of a driver
#             they want to get information on. The information
#             that is displayed is the number of wins in the last
#             two years & display the likelihood of the driver
#             winning at each track. You can also see the likelihood
#             of a driver winning the regular season and playoffs.
#CS-470
#Due Date: 04/28/2025
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Variables to keep track of for drivers
topFive2021 = topFive2022 = topFive2023 = topFive2024 = topFive2025 = 0
wins2021 = wins2022 = wins2023 = wins2024 = wins2025 = 0

#Ask for driver input
driver = input("Enter your driver: ").upper()

#Read data for previous seasons from excel file
file = pd.read_csv('NASCAR-2023-Stats.csv')
file.columns = file.columns.str.strip()
#Gather data for the driver the user input without checking for case sensitivity
driverData = file[file['Driver'].str.contains(driver, case=False, na=False)]



#Check for data while the file isn't at the end
if not driverData.empty:
    #Check for 2023 data that contains 2023 in it
    data2023 = driverData[driverData['Track/Year'].str.contains('2023', na=False)].copy()
    print(f"\n{driver}'s finishes in 2023:")#Display previous race results
    print(data2023[['Pos', 'St', 'Track/Year']].reset_index(drop=True).to_string(index=False))
    position2023 = data2023['Pos'].values#keep track of data in the Pos column in excel file
#Check for top fives and wins
    for pos in position2023:
        if pos <= 5:
            topFive2023 += 1
            if pos == 1:
                wins2023 += 1

#===Check above for explaination of code===

#Check for 2023 data that contains 2024 in it
data2024 = driverData[driverData['Track/Year'].str.contains('2024', na=False)].copy()
if not data2024.empty:
    print(f"\n{driver}'s finishes in 2024:")
    print(data2024[['Pos', 'St', 'Track/Year']].reset_index(drop=True).to_string(index=False))
    position2024 = data2024['Pos'].values
    for pos in position2024:
        if pos <= 5:
            topFive2024 += 1
            if pos == 1:
                wins2024 += 1

#Check for 2023 data that contains 2025 in it
currSeason = pd.read_csv('NASCAR2025-Season.csv')
driverCurrSeason = currSeason[currSeason['Driver'].str.contains(driver, case=False, na=False)]
data2025 = driverCurrSeason[driverCurrSeason['Track/Year'].str.contains('2025', na=False)].copy()
if not data2025.empty:
    print(f"\n{driver}'s finishes in 2025:")
    print(data2025[['Pos', 'St', 'Track/Year']].reset_index(drop=True).to_string(index=False))
    position2025 = data2025['Pos'].values
    for pos in position2025:
        if pos <= 5:
            topFive2025 += 1
            if pos == 1:
                wins2025 += 1

#Display all season stat summaries
print(f"\nTop fives for {driver} in 2023: {topFive2023}")
print(f"Wins for {driver} in 2023: {wins2023}")
print(f"Top fives for {driver} in 2024: {topFive2024}")
print(f"Wins for {driver} in 2024: {wins2024}")
print(f"Top fives for {driver} in 2025: {topFive2025}")
print(f"Wins for {driver} in 2025: {wins2025}")


#=== Graphing: Top Fives and Wins ===
#Make lists for the years and statistics
years = ['2023', '2024', '2025']
top_fives = [topFive2023, topFive2024, topFive2025]
wins = [wins2023, wins2024, wins2025]

#Plot Top Fives
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(years, top_fives, color='skyblue')
plt.title(f'{driver} - Top Five Finishes by Year')
plt.xlabel('Year')
plt.ylabel('Top Five Finishes')
plt.ylim(0, max(top_fives) + 1)

#Plot Wins
plt.subplot(1, 2, 2)
plt.bar(years, wins, color='orange')
plt.title(f'{driver} - Wins by Year')
plt.xlabel('Year')
plt.ylabel('Wins')
plt.ylim(0, max(wins) + 1)

plt.tight_layout()
plt.show()

#=== Machine Learning for Prediction ===

#Add helper columns for model training
#for-loop to go through each years dataset
for df, year in [(data2023, 2023), (data2024, 2024), (data2025, 2025)]:
    #Datafile gets a year column and adds it for each year
    df.loc[:, 'Year'] = year
    #Extract the track name from the year (x)
    #lambda will take track only because its in the first position of the column
    df.loc[:, 'Track'] = df['Track/Year'].apply(lambda x: x.split()[0])
    df.loc[:, 'Driver'] = df['Driver'].str.upper()#Make driver name all caps for uniformality

#Combine all years into one dataset
full_data = pd.concat([data2023, data2024, data2025], ignore_index=True)

#===Create feature matrix (X) and target vector (y)===

#Makes a matrix with all data
X = full_data[['St', 'Year', 'Track', 'Driver']]
#One-hot encoding to change these strings into numerical data
X = pd.get_dummies(X, columns=['Track', 'Driver'], drop_first=True)
y = full_data['Pos']#Target (y) is now the finishing position (continuous)

#===Train/test split and model training===

#Train 80% and Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Makes 100 trees and a fixed seed for reproducibility (42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluate the model using Mean Squared Error (MSE) and R^2
y_pred = model.predict(X_test)
#Thank God for statistics folks for making this for me
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)#Testing how well the model fits the data
print(f"\nModel Mean Squared Error: {mse:.2f}")
print(f"Model R^2: {r2:.2f}")

#=== Predict next race ===

#Ask user for next track and starting position
next_track = input("\nEnter the name of the next track (e.g., Daytona, Talladega): ").strip().title()
try:
    next_start_pos = int(input(f"Enter {driver}'s expected starting position at {next_track}: "))
except ValueError: #If user gives invalid entry give starting position=10
    print("Invalid input. Using default starting position of 10.")
    next_start_pos = 10

#Prepare prediction input
future_race = pd.DataFrame([{
    'St': next_start_pos,
    'Year': 2025,
    'Track': next_track,
    'Driver': driver
}])

#One-hot encode prediction input to match training data
future_race = pd.get_dummies(future_race)
future_race = future_race.reindex(columns=X.columns, fill_value=0)

#Predict finishing position and output
pred_position = model.predict(future_race)[0]
print(f"\nPrediction for {driver} at {next_track} 2025 (starting position {next_start_pos}):")
print(f"Likely to win? {'Yes' if pred_position == 1 else 'Maybe' if pred_position <= 15 else 'No'}")
print(f"Predicted finishing position: {pred_position:.2f}")
