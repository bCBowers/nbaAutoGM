# -*- coding: utf-8 -*-

# Import Packages and Data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from gurobipy import *

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

lineups_5_df = pd.read_csv('..\Git\data\lineups_5.csv.xz')
offense_df = pd.read_csv('..\Git\data\offense.csv.xz')
defense_df = pd.read_csv('..\Git\data\defense.csv.xz')

# Add mean heights and weights to lineup data
offense_df = offense_df.drop(['COLLEGE', 'COUNTRY', 'DRAFT_NUMBER', 'DRAFT_ROUND', 'DRAFT_YEAR', 'PLAYER_HEIGHT'], axis=1)
offense_df['PLAYER_ID'] = offense_df['PLAYER_ID'].astype(str)
shortlineups = lineups_5_df[['GROUP_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'SEASON']]
lineups_to_players = shortlineups['GROUP_ID'] \
    .str \
    .extractall(r'\b(?P<PLAYER_ID>\d+)\b')
lineups_to_players = shortlineups.reindex(lineups_to_players.index, level=0).join(lineups_to_players)
lineups_offense = pd.merge(lineups_to_players, offense_df, how='inner', on=['SEASON', 'PLAYER_ID'])
lineups_height = lineups_offense[['GROUP_ID', 'SEASON', 'PLAYER_HEIGHT_INCHES', 'PLAYER_WEIGHT']]
lineups_height = lineups_height.groupby(['GROUP_ID', 'SEASON']).agg({'PLAYER_HEIGHT_INCHES': 'mean', 
                                     'PLAYER_WEIGHT': 'mean'})
lineups_height = lineups_height.reset_index()
lineups_height = lineups_height.rename(columns={('GROUP_ID',''): 'GROUP_ID', ('SEASON',''): 'SEASON'})
lineups_5_df = pd.merge(lineups_5_df, lineups_height, how='inner', on=['SEASON', 'GROUP_ID'])

# Remove 2017-18 season
lineups_5_df= lineups_5_df[lineups_5_df['SEASON'] != '2017-18']

# Convert raw lineup totals to per minutes <- to compare across different minutes
lineups_5_df['FGM'] = lineups_5_df['FGM']/lineups_5_df['MIN']
lineups_5_df['FGA'] = lineups_5_df['FGA']/lineups_5_df['MIN']
lineups_5_df['FG3M'] = lineups_5_df['FG3M']/lineups_5_df['MIN']
lineups_5_df['FG3A'] = lineups_5_df['FG3A']/lineups_5_df['MIN']
lineups_5_df['FG2M'] = lineups_5_df['FGM'] - lineups_5_df['FG3M']
lineups_5_df['FG2A'] = lineups_5_df['FGA'] - lineups_5_df['FG3A']
lineups_5_df['FTM'] = lineups_5_df['FTM']/lineups_5_df['MIN']
lineups_5_df['FTA'] = lineups_5_df['FTA']/lineups_5_df['MIN']
lineups_5_df['OREB'] = lineups_5_df['OREB']/lineups_5_df['MIN']
lineups_5_df['DREB'] = lineups_5_df['DREB']/lineups_5_df['MIN']
lineups_5_df['REB'] = lineups_5_df['REB']/lineups_5_df['MIN']
lineups_5_df['AST'] = lineups_5_df['TOV']/lineups_5_df['MIN']
lineups_5_df['STL'] = lineups_5_df['STL']/lineups_5_df['MIN']
lineups_5_df['BLK'] = lineups_5_df['BLK']/lineups_5_df['MIN']
lineups_5_df['BLKA'] = lineups_5_df['BLKA']/lineups_5_df['MIN']
lineups_5_df['PF'] = lineups_5_df['PF']/lineups_5_df['MIN']
lineups_5_df['PFD'] = lineups_5_df['PFD']/lineups_5_df['MIN']
lineups_5_df['PTS'] = lineups_5_df['PTS']/lineups_5_df['MIN']
lineups_5_df['3PG'] = lineups_5_df['FG3M'] * (lineups_5_df['FG3M']/lineups_5_df['FG3A'])

# Keep useful fields
lineups_5_df = lineups_5_df[['GROUP_ID', 'OFF_RATING','SEASON', 'DEF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'TS_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS', '3PG', 'FG2M', 'FG2A']]

#Set up test and training data
lineups_train = lineups_5_df[lineups_5_df['SEASON'] != '2016-17']
lineups_test = lineups_5_df[lineups_5_df['SEASON'] == '2016-17']

# Set Index to Group ID
lineups_train = lineups_train.set_index(['GROUP_ID', 'SEASON'])
lineups_test = lineups_test.set_index(['GROUP_ID', 'SEASON'])

# Remove Ratings for Normalization
train_rating = lineups_train[['OFF_RATING', 'DEF_RATING']]
test_rating = lineups_test[['OFF_RATING', 'DEF_RATING']]
lineups_train = lineups_train.drop(['OFF_RATING', 'DEF_RATING'], axis=1)
lineups_test = lineups_test.drop(['OFF_RATING', 'DEF_RATING'], axis=1)

# Normalize code
lineups_train = (lineups_train - lineups_train.min()) / (lineups_train.max() - lineups_train.min())
lineups_test = (lineups_test - lineups_test.min()) / (lineups_test.max() - lineups_test.min())

lineups_train = lineups_train.reset_index()
lineups_test = lineups_test.reset_index()
train_rating = train_rating.reset_index()
test_rating = test_rating.reset_index()

lineups_train = pd.merge(lineups_train, train_rating, how='inner', on=['GROUP_ID', 'SEASON'])
lineups_test = pd.merge(lineups_test, test_rating, how='inner', on=['GROUP_ID', 'SEASON'])

# Set Index to Group ID
lineups_train = lineups_train.set_index(['GROUP_ID', 'SEASON'])
lineups_test = lineups_test.set_index(['GROUP_ID', 'SEASON'])


## Offense Lineup Model
# Build full LM
lineups_off_train = lineups_train[['OFF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS', '3PG', 'FG2M', 'FG2A']]

X = lineups_off_train
X = sm.add_constant(X)
X = lineups_off_train.drop(('OFF_RATING'), axis=1)
Y = lineups_off_train[['OFF_RATING']]
model = sm.OLS(Y, X).fit()
model.summary()

# Test for collinearity
X = lineups_off_train.drop(['OFF_RATING',  'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                            'PLAYER_WEIGHT', 'STL', 'BLK', 'BLKA', 
                            'PF', 'PFD'], axis=1)
corr = np.corrcoef(X, rowvar=0)  # correlation matrix

# Build LM of significant factors and remove collinear factors
X = lineups_off_train.drop(['OFF_RATING',  'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                            'PLAYER_WEIGHT', 'STL', 'BLK', 'BLKA', 
                            'PF', 'PFD', 'EFG_PCT', 'FGA', 'PTS',
                            '3PG', 'AST_TO', 'DREB_PCT', 'OREB_PCT',
                            'DREB', 'OREB', 'FTM', 'AST_PCT', 'TM_TOV_PCT', 
                            'REB_PCT', 'FG3M', 'FG2M', 'FG2A'], axis=1)
X = sm.add_constant(X)
Y = lineups_off_train[['OFF_RATING']]
off_lineup_model = sm.OLS(Y, X).fit()
off_lineup_model.summary()
lineups_off_test = lineups_test[['OFF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS', '3PG', 'FG2M', 'FG2A']]
x = lineups_off_test.drop(['OFF_RATING',  'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                            'PLAYER_WEIGHT', 'STL', 'BLK', 'BLKA', 
                            'PF', 'PFD', 'EFG_PCT', 'FGA', 'PTS',
                            '3PG', 'AST_TO', 'DREB_PCT', 'OREB_PCT',
                            'DREB', 'OREB', 'FTM', 'AST_PCT', 'TM_TOV_PCT', 
                            'REB_PCT', 'FG3M', 'FG2M', 'FG2A'], axis=1)
x = sm.add_constant(x)
y = lineups_off_test[['OFF_RATING']]
predictions = off_lineup_model.predict(x)
r2 = np.corrcoef(y['OFF_RATING'], predictions)[0,1]


'''
###### Demonstrating TS% Model overfit
lineups_off_train = lineups_train[['OFF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS', '3PG', 'FG2M', 'FG2A', 'TS_PCT', 'AVG_MADE_DISTANCE']]

X = lineups_off_train.drop(['OFF_RATING',  'PACE', 'PIE', 'PTS', 'AST',
                            'PLAYER_WEIGHT', 'STL', 'BLK', 'BLKA', 
                            'PF', 'PFD', 'EFG_PCT', 'FGA', 'PTS',
                            '3PG', 'AST_TO', 'DREB_PCT', 'REB', 'FTA',
                            'DREB', 'OREB', 'FTM', 'AST_PCT', 'FGM', 
                            'REB_PCT', 'FG3M', 'FG2M', 'FG2A', 'FG3A'], axis=1)
X = sm.add_constant(X)
Y = lineups_off_train[['OFF_RATING']]
off_lineup_model = sm.OLS(Y, X).fit()
off_lineup_model.summary()

# Build RF of significant factors
rf.fit(X, Y['OFF_RATING'])
predictions = rf.predict(x)
rf.score(x,y['OFF_RATING'])
r2 = np.corrcoef(y['OFF_RATING'], predictions)[0,1]

lineups_off_test = lineups_test[['OFF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS', '3PG', 'FG2M', 'FG2A', 'TS_PCT', 'AVG_MADE_DISTANCE']]
x = lineups_off_test.drop(['OFF_RATING',  'PACE', 'PIE', 'PTS', 'AST',
                            'PLAYER_WEIGHT', 'STL', 'BLK', 'BLKA', 
                            'PF', 'PFD', 'EFG_PCT', 'FGA', 'PTS',
                            '3PG', 'AST_TO', 'DREB_PCT', 'REB', 'FTA',
                            'DREB', 'OREB', 'FTM', 'AST_PCT', 'FGM', 
                            'REB_PCT', 'FG3M', 'FG2M', 'FG2A', 'FG3A'], axis=1)
x = sm.add_constant(x)
y = lineups_off_test[['OFF_RATING']]
predictions = off_lineup_model.predict(x)
r2 = np.corrcoef(y['OFF_RATING'], predictions)[0,1]
          
a = np.arange (90, 125, 1)
z = np.arange (90, 125, 1)
plt.plot(y['OFF_RATING'], predictions, 'ro', a, z, 'g--')
plt.ylabel('Predicted Offensive Rating'); plt.xlabel('True Offensive Rating'); plt.title('Offensive Rating Accuracy')

offense_df = pd.read_csv('..\Git\data\offense.csv.xz')
playerVer = offense_df
playerVer  = playerVer[playerVer['MIN'] > 200]

playerVer = playerVer[['PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'TS_PCT', 'OREB_PCT', 'TM_TOV_PCT', 'PLAYER_HEIGHT_INCHES']]


# playerVer = playerVer[playerVer['SEASON'] != '2017-18']
# Offense rating predictions
X = playerVer[['PLAYER_NAME', 'OREB_PCT', 'TM_TOV_PCT', 'PLAYER_HEIGHT_INCHES', 'TS_PCT', 'AVG_MADE_DISTANCE']]
X = X.set_index(['PLAYER_NAME'])

X = (X - X.min()) / (X.max() - X.min())
X = sm.add_constant(X)
off_pred = off_lineup_model.predict(X)

playerVer['Off_Pred'] = off_pred
playerVer18 = playerVer[playerVer['SEASON'] == '2017-18']

playerVer18.to_csv('ver18.csv')
'''

'''
# Build RF of significant factors
rf.fit(X, Y['OFF_RATING'])
predictions = rf.predict(x)
rf.score(x,y['OFF_RATING'])
r2 = np.corrcoef(y['OFF_RATING'], predictions)[0,1]

importance = rf.feature_importances_
importance = importance.tolist()
index= X.T.index.tolist()
imp={}
for i in range(0,len(index)):
    imp[index[i]] = importance[i]
sorted_x = sorted(imp.items(), key=operator.itemgetter(1))
plt.plot(y['OFF_RATING'], predictions, 'ro', a, z, 'g--')
plt.ylabel('Predicted Offensive Rating'); plt.xlabel('True Offensive Rating'); plt.title('Offensive Rating Accuracy')

# Make visualization of Variable Importance
feature_list = list(x.columns)
feature_list = map(lambda x:x if x!= 'PLAYER_HEIGHT_INCHES' else 'HEIGHT',feature_list)
plt.style.use('fivethirtyeight')
x_values = list(range(len(importance)))
plt.bar(x_values, importance, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Offensive Variable Importances')
'''

## Defense Lineup Model
# Build full LM
lineups_def_train = lineups_train[['DEF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'TS_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS']]

X = lineups_def_train.drop(('DEF_RATING'), axis=1)
X = sm.add_constant(X)
Y = lineups_def_train[['DEF_RATING']]
model = sm.OLS(Y, X).fit()
model.summary()

# Test for collinearity
X = lineups_def_train.drop(['DEF_RATING', 'AST_TO', 'AST_PCT', 'REB_PCT', 'EFG_PCT',
                            'PACE', 'PIE', 'PLAYER_WEIGHT',
                            'FGM', 'FG3M', 'FGA', 'FG3A', 'FTM', 'FTA', 'AST', 'PTS'], axis=1)
corr = np.corrcoef(X, rowvar=0)  # correlation matrix

# Build LM of significant factors and remove collinear factors
X = lineups_def_train.drop(['DEF_RATING', 'AST_TO', 'AST_PCT', 'REB_PCT', 'EFG_PCT',
                            'PACE', 'PIE', 'PLAYER_WEIGHT',
                            'FGM', 'FG3M', 'FGA', 'FG3A', 'FTM', 'FTA', 'AST', 'PTS', 
                            'OREB_PCT', 'DREB_PCT', 'OREB', 'PFD', 'TS_PCT', 'BLKA', 
                            'TM_TOV_PCT', 'DREB'], axis=1)
X = sm.add_constant(X)
Y = lineups_def_train[['DEF_RATING']]
def_lineup_model = sm.OLS(Y, X).fit()
predictions = def_lineup_model.predict(X)
def_lineup_model.summary()

lineups_def_test = lineups_test[['DEF_RATING', 'AST_PCT', 'AST_TO',
                                   'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
                                   'EFG_PCT', 'TS_PCT', 'PACE', 'PIE', 'PLAYER_HEIGHT_INCHES',
                                   'PLAYER_WEIGHT', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                                   'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                                   'PTS']]
x = lineups_def_test.drop(['DEF_RATING', 'AST_TO', 'AST_PCT', 'REB_PCT', 'EFG_PCT',
                            'PACE', 'PIE', 'PLAYER_WEIGHT',
                            'FGM', 'FG3M', 'FGA', 'FG3A', 'FTM', 'FTA', 'AST', 'PTS', 
                            'OREB_PCT', 'DREB_PCT', 'OREB', 'PFD', 'TS_PCT', 'BLKA', 
                            'TM_TOV_PCT', 'DREB'], axis=1)
x = sm.add_constant(x)
y = lineups_def_test[['DEF_RATING']]
predictions = def_lineup_model.predict(x)
r2 = np.corrcoef(y['DEF_RATING'], predictions)[0,1]

# Build plot
a = np.arange (90, 125, 1)
z = np.arange (90, 125, 1)
plt.plot(y['DEF_RATING'], predictions, 'ro', a, z, 'g--')
plt.ylabel('Predicted Defensive Rating'); plt.xlabel('True Defensive Rating'); plt.title('Defensive Rating Accuracy')

'''
# Build RF of significant factors
rf.fit(X, Y['DEF_RATING'])
predictions = rf.predict(x)
rf.score(x,y['DEF_RATING'])
r2 = np.corrcoef(y['DEF_RATING'], predictions)[0,1]
importance = rf.feature_importances_
importance = importance.tolist()
index= X.T.index.tolist()
imp={}
for i in range(0,len(index)):
    imp[index[i]] = importance[i]
sorted_x = sorted(imp.items(), key=operator.itemgetter(1))
plt.plot(y['DEF_RATING'], predictions, 'ro', a, z, 'g--')
plt.ylabel('Predicted Defensive Rating'); plt.xlabel('True Defensive Rating'); plt.title('Defensive Rating Accuracy')


# Make visualization of Variable Importance
feature_list = list(x.columns)
feature_list = map(lambda x:x if x!= 'PLAYER_HEIGHT_INCHES' else 'HEIGHT',feature_list)
plt.style.use('fivethirtyeight')
x_values = list(range(len(importance)))
plt.bar(x_values, importance, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Defensive Variable Importances')

'''
offense_df = pd.read_csv('..\Git\data\offense.csv.xz')
defense_df = pd.read_csv('..\Git\data\defense.csv.xz')
defense_df = defense_df[['PLAYER_ID', 'SEASON', 'STL', 'BLK', 'PF']]
playerVer = pd.merge(offense_df, defense_df, how='inner', on=['PLAYER_ID', 'SEASON'])
playerVer  = playerVer[playerVer['MIN'] > 200]

playerVer = playerVer[['PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'AST', 'FGM', 'FTA', 
                       'REB', 'FG3A',  'STL', 'BLK', 'PF', 'PLAYER_HEIGHT_INCHES']]

# Offense rating predictions
X = playerVer[['PLAYER_NAME', 'FGM', 'FG3A', 'FTA', 'REB', 'AST']]
X = X.set_index(['PLAYER_NAME'])

X = (X - X.min()) / (X.max() - X.min())
X = sm.add_constant(X)
off_pred = off_lineup_model.predict(X)

X = playerVer[['PLAYER_HEIGHT_INCHES', 'REB',  'STL', 'BLK', 'PF']]
X = (X - X.min()) / (X.max() - X.min())
X = sm.add_constant(X)
def_pred = def_lineup_model.predict(X)

playerVer['Off_Pred'] = off_pred
playerVer['Def_Pred'] = def_pred
playerVer['Net'] = off_pred - def_pred
         
#### Apply the salary model and optimize
cand = pd.read_csv('WizCand.csv')
playerVer18 = playerVer[playerVer['SEASON'] == '2017-18']
wizCand = pd.merge(cand, playerVer18, how='inner', on=['PLAYER_ID'])
         
# 2018-19 Wizards team w/ Devin Harris (winner of optimization)
playerVer18 = playerVer[playerVer['SEASON'] == '2017-18']
playerVer18 = playerVer18.set_index(['PLAYER_ID'])
playerVerWiz = playerVer18.loc[[202322, 203078, 201975, 203107, 203490, 1626162, 202693,
                                201160, 101133, 101162, 2734]]

playerVer18.to_csv('ver18.csv')
# Lineup Optimization
optFile = pd.read_csv('optFile.csv')
rating = playerVerWiz[['Net']]
rating = rating.reset_index()
optFile= pd.merge(optFile, rating, how='inner', on=['PLAYER_ID'])
optFile = optFile.set_index(['PLAYER_ID', 'LINEUP']).T.to_dict()

Wiz = Model()
Wiz.modelSense = GRB.MAXIMIZE
Wiz.update()

# Set objective function
lineups = {}
rating = playerVerWiz[['Net']]

lines = [1, 2, 3, 4, 5, 6, 7]
line_time = {1: 15, 2: 10, 3: 8, 4: 5, 5: 5, 6: 3, 7: 2}
players = [202322, 203078, 201975, 203107, 203490, 1626162, 202693, 201160, 101133, 101162, 2734]

for p, l in optFile:
    cname = 'x(%s_%s)' % (p, l)
    lineups[p, l] = Wiz.addVar(obj = optFile[p,l]['Net'], lb=  0, 
                                   vtype = GRB.BINARY,
                                   name = cname)
# Define Constraints
myConstrDict={}

for l in lines:
    cName = '01_Five_Players_per_Lineup_%s' % (l)
    myConstrDict[cName] = Wiz.addConstr(quicksum(lineups[p,l] for p in players) == 5, name = cName)
Wiz.update() 

for p in players:
    cName = '02_Limit_Game_Time_%s' % (p)
    myConstrDict[cName] = Wiz.addConstr(quicksum(lineups[p,l] * line_time[l] for l in lines) <= 35, name = cName)
Wiz.update() 


# Check Model
Wiz.write('mylp.lp')

Wiz.optimize()

if Wiz.Status == GRB.OPTIMAL:
    wizLine = []
    for s in lineups:
        if lineups[s].x > 0.1: #0.1 instead of 0 to account for values slightly > 0
            wizLine.append((s[0], s[1], lineups[s].x))

wizLine = pd.DataFrame(wizLine, columns = ['PLAYER_ID', 'LINEUP', 'YES'])
playerVer18 = playerVer18.reset_index()
wizName = playerVer18[['PLAYER_ID', 'PLAYER_NAME']]

# Wizards by Minute per Game
wizLine = pd.merge(wizLine, wizName, how='inner', on=['PLAYER_ID'])
line_time = [[1, 15], [2, 10], [3, 8], [4, 5], [5, 5], [6, 3], [7, 2]]
line_time = pd.DataFrame(line_time, columns=['LINEUP', 'Time'])

wizLine = pd.merge(wizLine, line_time, how='inner', on=['LINEUP'])
wizLine = wizLine.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg({'Time': 'sum'})
wizLine = wizLine.reset_index()
rating = rating.reset_index()
wizLine = pd.merge(wizLine, rating, how='inner', on=['PLAYER_ID'])

# Offensive Rating Histogram
plt.hist(playerVer.Off_Pred, color='blue')
plt.title("Offensive Rating Histogram")
plt.xlabel("Offensive Rating")
plt.ylabel("Frequency")
plt.gcf()
np.mean(playerVer.Off_Pred)
np.std(playerVer.Off_Pred)
plt.close()

# Defensive Rating Histogram
playerVer = playerVer[playerVer['PLAYER_ID']!=202399]
plt.hist(playerVer.Def_Pred, color='red')
plt.title("Defensive Rating Histogram")
plt.xlabel("Defensive Rating")
plt.ylabel("Frequency")
plt.gcf()
np.mean(playerVer.Def_Pred)
np.std(playerVer.Def_Pred)
plt.close()

# Net Rating Histogram
plt.hist(playerVer.Net, color='green')
plt.title("Net Rating Histogram")
plt.xlabel("Net Rating")
plt.ylabel("Frequency")
plt.gcf()
np.mean(playerVer.Net)
np.std(playerVer.Net)


'''
# Extract 2016-17 season
playerVer16 = playerVer[playerVer['SEASON'] == '2016-17']
defVerif = pd.read_csv('defVerif.csv')

playerVer16 = playerVer16[['PLAYER_ID', 'PLAYER_NAME', 'Off_Pred']]
defVerif = defVerif[['PLAYER_ID', 'PLAYER_NAME', 'Def_Pred']]

playerVer16 = pd.merge(playerVer16, defVerif, how='inner', on=['PLAYER_ID', 'PLAYER_NAME'])
playerVer16['Net_Pred'] = playerVer16['Off_Pred'] - playerVer16['Def_Pred']

playerVer.to_csv('playerVerif.csv')


#### Produce visualization of Player Rating
# Offensive Rating Histogram
plt.hist(playerVer16.Off_Pred, color='blue')
plt.title("Predicted Player Offensive Rating Histogram")
plt.xlabel("Offensive Rating")
plt.ylabel("Frequency")
plt.gcf()
np.mean(playerVer16.Off_Pred)
np.std(playerVer16.Off_Pred)
plt.close()

# Defensive Rating Histogram
plt.hist(playerVer16.Def_Pred, color='red')
plt.title("Predicted Player Defensive Rating Histogram")
plt.xlabel("Defensive Rating")
plt.ylabel("Frequency")
plt.gcf()
np.mean(playerVer16.Def_Pred)
np.std(playerVer16.Def_Pred)
plt.close()

# Net Rating Histogram
plt.hist(playerVer16.Net_Pred, color='green')
plt.title("Predicted Player Net Rating Histogram")
plt.xlabel("Net Rating")
plt.ylabel("Frequency")
plt.gcf()
np.mean(playerVer16.Net_Pred)
np.std(playerVer16.Net_Pred)


'''
