import json
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix

field_height = 53.3
field_length = 120
endzone_length = 10
seconds_in_a_half = 15 * 60
seconds_in_overtime = 10 * 60

def get_data(x_file=None, y_file=None):
    if x_file and y_file:
        with open(x_file) as f:
            x = np.array(json.load(f))
        with open(y_file) as f:
            y = np.array(json.load(f))
        return x, y

    games = pd.read_csv('data/games.csv')
    plays = pd.read_csv('data/plays.csv')
    plays = plays.merge(games, on='gameId')
    plays['is_home_offense'] = plays['possessionTeam'] == plays['homeTeamAbbr']

    tracking_2018 = pd.read_csv('data/tracking2018.csv')
    tracking_2019 = pd.read_csv('data/tracking2019.csv')
    tracking_2020 = pd.read_csv('data/tracking2020.csv')
    tracking = tracking_2018.append([tracking_2019, tracking_2020])

    left_mask = tracking['playDirection'] == 'left'
    right_mask = tracking['playDirection'] == 'right'
    tracking.loc[left_mask, 'stdX'] = field_length - tracking.loc[left_mask, 'x']
    tracking.loc[left_mask, 'stdY'] = field_height - tracking.loc[left_mask, 'y']
    tracking.loc[left_mask, 'stdDir'] = (360 - tracking.loc[left_mask, 'dir']) % 360
    tracking.loc[right_mask, 'stdX'] = tracking.loc[right_mask, 'x']
    tracking.loc[right_mask, 'stdY'] = tracking.loc[right_mask, 'y']
    tracking.loc[right_mask, 'stdDir'] = tracking.loc[right_mask, 'dir']
    tracking['stdR'] = (tracking['stdX'].pow(2) + tracking['stdY'].pow(2)).pow(1/2)
    tracking['stdTheta'] = np.arctan(tracking['stdY']/tracking['stdX'])
    tracking['nflId'] = tracking['nflId'].fillna(-1)

    num_games = plays['gameId'].unique().size

    play_results = plays.groupby(['specialTeamsPlayType', 'specialTeamsResult']).size().unstack('specialTeamsPlayType')
    play_results = play_results.divide(num_games)
    play_results_probs = play_results.div(play_results.sum(axis=0), axis=1)
    excitement_scores = play_results_probs.subtract(1).multiply(-1).unstack().dropna().rename('excitement_score_v1')
    plays = plays.join(excitement_scores, on=['specialTeamsPlayType', 'specialTeamsResult'])

    plays.loc[plays['is_home_offense'], 'score_diff'] = plays['preSnapHomeScore'] - plays['preSnapVisitorScore']
    plays.loc[~plays['is_home_offense'], 'score_diff'] = plays['preSnapVisitorScore'] - plays['preSnapHomeScore']
    plays['score_diff_ex'] = pd.cut(plays['score_diff'].abs(), 
        bins=[-1, 0, 1, 3, 5, 8, 11, 16, 21, 26, 31], 
        labels=[.9, 1, .8, .7, .6, .5, .4, .3, .2, .1]
    ).astype(float)
    plays['excitement_score_v2'] = 0.5 * plays['excitement_score_v1'] + 0.5 * plays['score_diff_ex']

    plays['gameClockMins'] = plays['gameClock'].str.split(':').str[0]
    plays['gameClockSecs'] = plays['gameClock'].str.split(':').str[1]
    plays['seconds_left_in_quarter'] = plays['gameClockMins'].astype(int) * 60 + plays['gameClockSecs'].astype(int)
    plays.loc[plays['quarter'].isin([2, 4, 5]), 'seconds_left_in_half'] = plays['seconds_left_in_quarter']
    plays.loc[plays['quarter'].isin([1, 3]), 'seconds_left_in_half'] = plays['seconds_left_in_quarter'] + seconds_in_a_half
    plays.loc[plays['quarter'] != 5, 'time_left_ex'] = (1 - plays['seconds_left_in_half'] / (seconds_in_a_half * 2))
    plays.loc[plays['quarter'] == 5, 'time_left_ex'] = (1 - plays['seconds_left_in_half'] / seconds_in_overtime)
    plays['quarter_ex'] = plays['quarter'] / 5
    plays['excitement_score_v3'] = 0.25 * (plays['excitement_score_v1'] + plays['score_diff_ex'] + plays['quarter_ex'] + plays['time_left_ex'])

    kickoffs = plays[plays['specialTeamsPlayType']=='Kickoff'].reset_index(drop=True)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    transformed = encoder.fit_transform(kickoffs[['specialTeamsResult']])
    output_cols = [x.split('_')[1] for x in encoder.get_feature_names_out()]
    ohe_df = pd.DataFrame(transformed, columns=output_cols)
    kickoffs = pd.concat([kickoffs, ohe_df], axis=1)

    kickoffs['initial_excitement'] = 1/3 * (kickoffs['score_diff_ex'] + kickoffs['quarter_ex'] + kickoffs['time_left_ex'])
    kickoff_tracking = kickoffs.merge(tracking, on=['gameId', 'playId'])

    kickoff_tracking['is_off'] = (kickoff_tracking['is_home_offense'] & (kickoff_tracking['team'] == 'home')) | (~kickoff_tracking['is_home_offense'] & (kickoff_tracking['team'] == 'away')) | (kickoff_tracking['team'] == 'football')

    features = ['stdR', 'stdTheta']

    sc = MinMaxScaler()
    kickoff_tracking[features] = sc.fit_transform(kickoff_tracking[features])

    sort_cols = ['gameId', 'playId', 'frameId', 'is_off', 'nflId']
    df = kickoff_tracking[[*sort_cols, *features]].set_index(sort_cols).sort_index(level=sort_cols).droplevel('nflId')
    df = df.set_index(df.groupby(['gameId', 'playId', 'frameId']).cumcount().rename('playerNum'), append=True)
    df = df[features].unstack(['is_off', 'playerNum']).reset_index(['gameId', 'playId', 'frameId'])

    x = np.array([x[~np.isnan(x)] for x in df.values])
    x = np.array([x[3:] for x in x])

    y = kickoff_tracking.groupby(['gameId', 'playId', 'frameId'])[output_cols].first().values

    return x, y



x, y = get_data('x.json', 'y.json')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

num_input_nodes = x.shape[1]
num_hidden_nodes = x.shape[1] * 2
num_output_nodes = y.shape[1]

model = Sequential()
model.add(Dense(num_hidden_nodes, input_dim=num_input_nodes, activation='relu'))
model.add(Dense(num_output_nodes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, verbose=1)

results = model.evaluate(x_test, y_test)
print(results)

y_pred = model.predict(x_test)
for pred, actual in zip(y_pred[:10], y_test[:10]):
	print(f'Pred: {pred}; Actual: {actual}')

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)
