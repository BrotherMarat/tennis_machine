import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')







# perform exploratory analysis here:
#plt.scatter(df.BreakPointsOpportunities, df.Winnings)


income = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

income_train, income_test, outcome_train, outcome_test = train_test_split(income, outcome, train_size = 0.8, test_size = 0.2, random_state = 6)

tr = LinearRegression()

modeli = tr.fit(income_train, outcome_train)

outcome_predicted = tr.predict(income_test)

print(tr.score(income_train, outcome_train))
print(tr.score(income_test, outcome_test))

plt.scatter(outcome_test, outcome_predicted, alpha=0.4)
plt.plot(range(1000000), range(1000000))

plt.xlabel('Winnings $')
plt.ylabel('Predicted Winnings $')

plt.show()

new_player_test = [[0.9, 0.6, 0.3, 0.54, 0.44, 2, 0.0, 8, 5, 0.3, 1, 9, 0.15, 0.3, 16, 0.65, 0.45, 0.58]]

predict = modeli.predict(new_player_test)
print("Predicted winnigs: $%.2f" % predict)















## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
