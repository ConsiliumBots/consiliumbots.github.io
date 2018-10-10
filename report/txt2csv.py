import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import scipy, random, copy, h5py, pandas, math, csv, sys, os, pickle, re, json, warnings
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn import preprocessing
from numpy.random import RandomState
from scipy import ndimage
from scipy import stats
from scipy import ndimage
from datetime import datetime
path = os.getcwd()

sub_routines = os.path.join(path,'..','QueVasaEstudiar','python', 'sub_routines')
sys.path.append(sub_routines)
from neural_predictor import neural_predictor
from Import_MenuParameters import Import_MenuParameters
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

gcolor = '#e6550d'

#1. Import data and parameters:
root = "C:/Users/Franco/GitHub" #Franco's MSI

csv_path = root + "/Bots_Colombia/data/"

def loadOutcomes():
    outcomes = {}
    with open(root + '/QueVasaEstudiar/python/Options_Outcomes_05_15_2018.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            outcomes[int(row['Option_ID'])] = row
    return outcomes

outcomes = loadOutcomes()

def arange_outcomes(outcomes):

    HEOptions = {
    'MajorName'       : np.array([outcomes[x]['Option_Major_Name'] for x in outcomes.keys()]),
    'OptionID'        : np.array([x for x in outcomes.keys()]).astype(int),
    'InstCode'        : np.array([outcomes[x]['Option_Institution_Code'] for x in outcomes.keys()]).astype(int) -1,
    'Cutoff'          : np.array([outcomes[x]['LowerBoundScore'] for x in outcomes.keys()]).astype(float),
    'NoSearch'        : np.array([outcomes[x]['NoSearch'] for x in outcomes.keys()]).astype(int),
    'LevelCode'       : np.array([outcomes[x]['Option_Level_Code'] for x in outcomes.keys()]).astype(int) -1,
    'AreaCode'        : np.array([outcomes[x]['Option_Area_Code'] for x in outcomes.keys()]).astype(int) -1,
    'LocationCode'    : np.array([outcomes[x]['Option_Location_Code'] for x in outcomes.keys()]).astype(int) -1,
    'MajorCode'       : np.array([outcomes[x]['Option_Major_Code'] for x in outcomes.keys()]).astype(int) -1,
    'AverageEarning'  : np.array([outcomes[x]['Outcome_IncomeAve'] if outcomes[x]['Outcome_IncomeAve'] != "" else "-999" for x in outcomes.keys()]).astype(float)
    }

    HEOptions['AreaID']     = np.array(list(set(HEOptions['AreaCode'])))
    HEOptions['InstID']     = np.array(list(set(HEOptions['InstCode'])))      #We have to change this part
    HEOptions['LevelID']    = np.array(list(set(HEOptions['LevelCode'])))
    HEOptions['LocationID'] = np.array(list(set(HEOptions['LocationCode'])))
    HEOptions['MajorID']    = np.array(list(set(HEOptions['MajorCode'])))

    #Fixes:
    aux_InstID = np.array(range(HEOptions['InstID'].shape[0]))

    for y in HEOptions:
        HEOptions[y] = HEOptions[y][:,np.newaxis]

    for i in range(HEOptions['InstID'].shape[0]):
        HEOptions['InstCode'][HEOptions['InstCode'] == HEOptions['InstID'][i,0]] = aux_InstID[i]

    HEOptions['RelevantInst']  = aux_InstID[:,np.newaxis]
    HEOptions['normW']         = 10000000
    HEOptions['AverageEarning'][HEOptions['AverageEarning'] == -999] = np.nan

    return HEOptions

#1. SET UP AND IMPORT DATA:
#==========================

#1.1. IMPORT DATA ABOUT OPTIONS:
HEOptions = arange_outcomes(outcomes)

allinteractions = pd.read_csv(root+'/Bots_Colombia/ModelEstimation/deeplearning_estimation/allinteractions.csv', error_bad_lines=False)
allinteractions = allinteractions[(allinteractions.user != 'default-user') & (allinteractions.user.str.len() == 20)].reset_index(drop=True)
allinteractions["year"] = allinteractions.timestamp.str[0:4].astype(int)
allinteractions["month"] = allinteractions.timestamp.str[5:7].astype(int)
allinteractions["day"] = allinteractions.timestamp.str[8:10].astype(int)
allinteractions["second"] = allinteractions.timestamp.str[17:19].astype(int)
allinteractions["minute"] = allinteractions.timestamp.str[14:16].astype(int)
allinteractions["hour"] = allinteractions.timestamp.str[11:13].astype(int) - 7
allinteractions["hour"][allinteractions["hour"]==-7] = 17
allinteractions["hour"][allinteractions["hour"]==-6] = 18
allinteractions["hour"][allinteractions["hour"]==-5] = 19
allinteractions["hour"][allinteractions["hour"]==-4] = 20
allinteractions["hour"][allinteractions["hour"]==-3] = 21
allinteractions["hour"][allinteractions["hour"]==-2] = 22
allinteractions["hour"][allinteractions["hour"]==-1] = 23

allinteractions = allinteractions[(allinteractions.year == 2018) & (allinteractions.month >= 9)]
allinteractions = allinteractions[(allinteractions.user != "992") & (allinteractions.user != "YBOF9JBM8DE9ME6RE1K4") & (allinteractions.user != "EMF393CY0PX7ESW0BNQ")]
allinteractions = allinteractions.sort_values(['month','day'])
allinteractions['Cummulative'] = np.cumsum(np.ones((allinteractions.shape[0],1))).astype(int)
allinteractions['Interacted'] = 1
allinteractions['times'] = list(map(lambda x: x[0:10] + " " + x[11:19], allinteractions['timestamp']))
allinteractions['times'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), allinteractions['times']))
allinteractions['hours'] = list(map(lambda x: x[0:10] + " " + x[11:13], allinteractions['timestamp']))
allinteractions['hours'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H'), allinteractions['hours']))
allinteractions['days'] = list(map(lambda x: x[0:10] , allinteractions['timestamp']))
allinteractions['days'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), allinteractions['days']))


#1.3. IMPORT DATA ABOUT WAGE DEVIATIONS:
wageDeviation   = pd.read_csv(root+'/Bots_Colombia/ModelEstimation/deeplearning_estimation/wage_deviation.csv'.format(100)).drop(columns = ['Unnamed: 0']).rename(columns= {'studentID': 'user'})
wageDeviation['hours'] = list(map(lambda x: x[0:10] + " " + x[11:13], wageDeviation['time']))
wageDeviation['hours'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H'), wageDeviation['hours']))
wageDeviation['times'] = list(map(lambda x: x[0:10] + " " + x[11:19], wageDeviation['time']))
wageDeviation['times'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), wageDeviation['times']))
wageDeviation['days'] = list(map(lambda x: x[0:10] , wageDeviation['time']))
wageDeviation['days'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), wageDeviation['days']))
wageDeviation['interaction'] = 1
wageDeviation['absWageDeviation'] = list(map(lambda x: abs(x), wageDeviation['wageDeviation']))

i = 1
j = 0
wageDeviation = wageDeviation.sort_values(by = ['user', 'times'])
wageDeviation['numberMenu'] = -999
for i in range(1,wageDeviation.shape[0]):
    if wageDeviation['user'].iloc[i] == wageDeviation['user'].iloc[i-1]:
        j += 1
        wageDeviation['numberMenu'].iloc[i] = j
    else:
        j = 1
        wageDeviation['numberMenu'].iloc[i] = j

wageDeviation['numberMenu'].iloc[0]=1
wageDeviation['numberMenu'][wageDeviation['numberMenu'] == 8] = 8
wageDeviation['numberMenu'][wageDeviation['numberMenu'] == 9] = 8
wageDeviation['numberMenu'][wageDeviation['numberMenu']>=10] = 9
wageDeviation['N_total']=1

wageDeviation['Max_interaction'] = ""
wageDeviation = wageDeviation.merge(wageDeviation.groupby(['user']).sum().reset_index()[['user', 'interaction']].rename(columns ={'interaction':'max_interaction'}), left_on = 'user', right_on = 'user')
wageDeviation['Max_interaction'][(wageDeviation['max_interaction'] > 0) & (wageDeviation['max_interaction'] <= 3)] = ']0-3]'
wageDeviation['Max_interaction'][(wageDeviation['max_interaction'] > 3) & (wageDeviation['max_interaction'] <= 6)] = ']3-6]'
wageDeviation['Max_interaction'][(wageDeviation['max_interaction'] > 6) & (wageDeviation['max_interaction'] <= 9)] = ']6-9]'
wageDeviation['Max_interaction'][wageDeviation['max_interaction'] >= 9] = ']9-inf]'



#1. Data for performance of the Structural and ML model:
#======================================================

parameters = {}
path_pickle = os.path.join(path, '..', 'Bots_Colombia', 'ModelEstimation','deeplearning_estimation', '')
with open(path_pickle + 'parameters_nn.pickle', 'rb') as handle:
    parameters['NN'] = pickle.load(handle)

studentFeatures = pd.read_csv(root+'/QueVasaEstudiar/Student_Features_Fall2018.csv'.format(100))
interactionsBot = pd.read_csv(root+'/Bots_Colombia/ModelEstimation/deeplearning_estimation/interactions_bot.csv'.format(100))
seedAssignment  = pd.read_csv(root+'/Bots_Colombia/ModelEstimation/deeplearning_estimation/seeds_intervention.csv').drop(columns=['Unnamed: 0'])
interactions    = seedAssignment.merge(studentFeatures, left_on = 'url_id', right_on = 'url_id', how = 'left')
interactions    = interactions.merge(interactionsBot, left_on = 'Student_ID', right_on = 'Student_ID', how = 'inner')

n_int = interactions.shape[0]
n_options = HEOptions['MajorName'].shape[0]

#2.2. Generate tags
dpto_code = list(set(studentFeatures['Student_Location_Code']))

for x in dpto_code:
    interactions['dpto_'+str(x)] = (interactions['Student_Location_Code'] == x).astype(int)
    dpto_tag = ['dpto_'+str(x) for x in dpto_code]

menu_tag = ['Menu1','Menu2','Menu3','Menu4']
tag_continuous = ['punt_rez_sede','inse_rez_sede','puntsd_rez_sede', 'insesd_rez_sede', 'edad']

X_indexslates = np.array(interactions[menu_tag])
for i,j in zip(range(n_options), list(HEOptions['OptionID'][:,0])):
    X_indexslates[:,0][X_indexslates[:,0]==j] = i
    X_indexslates[:,1][X_indexslates[:,1]==j] = i
    X_indexslates[:,2][X_indexslates[:,2]==j] = i
    X_indexslates[:,3][X_indexslates[:,3]==j] = i

X_SlateEarnigs          = HEOptions['AverageEarning'][X_indexslates].reshape(n_int,len(menu_tag))
X_SlateEarnigs[np.isnan(X_SlateEarnigs)] = np.nanmean(X_SlateEarnigs)
X_SlateEarnigs          = X_SlateEarnigs
X_SlateLevels           = HEOptions['LevelCode'][X_indexslates].reshape(n_int,len(menu_tag))
X_SlateInst             = HEOptions['InstCode'][X_indexslates].reshape(n_int,len(menu_tag))
X_SlateArea             = HEOptions['AreaCode'][X_indexslates].reshape(n_int,len(menu_tag))
X_Score                 = np.array(interactions['Student_Score']).reshape(n_int,1)
X_ScoreDecil            = np.array(interactions['Student_ScoreDecil']).reshape(n_int,1)
X_Continuous            = np.array(interactions[tag_continuous]).reshape(n_int,len(tag_continuous)).astype(int)
X_Gender                = np.array(interactions['Student_Gender']).reshape(n_int,1)
X_LocationCode          = np.array(interactions[dpto_tag]).reshape(n_int,len(dpto_tag)).astype(int)

X                       = np.hstack((X_SlateEarnigs,X_SlateLevels,X_SlateInst,X_SlateArea,X_Gender,X_Score,X_ScoreDecil,X_LocationCode,X_Continuous))
Y                       = np.array(interactions['Selection']).reshape(n_int, 1) == np.concatenate((np.array(interactions[['Menu1','Menu2','Menu3','Menu4']]), np.array([-999]*n_int).reshape(n_int,1)), axis =  1)

X                       = (X - parameters['NN']['X_mean'])/parameters['NN']['X_std']


ClickProb, _ = neural_predictor(X.T, parameters['NN'], 'NN', None, None)
Y_hat = ClickProb.T == ClickProb.T.max(1).reshape(n_int,1)

#2. How does the neural network perform on the data provided by the different treatment arms??

#2.1. Accuracy
Accuracy = (Y * Y_hat).sum(1).reshape(n_int,1)

#2.2. Probability
Pr = (ClickProb.T * (Y * Y_hat)).sum(1).reshape(n_int,1)

#2.3. Expected Value:
Pr_dist = ClickProb.T[:,:-1]/ClickProb.T[:,:-1].sum(1).reshape(n_int,1)
EV = (X_SlateEarnigs*Pr_dist).sum(1).reshape(n_int,1)/10**7

#3. Compute cool graphics:
#3.1 Create DataFrame that allows us to create cool graphics:
TreatmentArm = np.array(interactions['bot']).reshape(n_int,1)
NeuralNet_DataFrame = pd.DataFrame(np.hstack((TreatmentArm, Accuracy, Pr, EV)), columns = ['TreatmentArm', 'Accuracy','Probability','ExpectedValue']).astype(float)
NeuralNet_DataFrame.TreatmentArm = NeuralNet_DataFrame.TreatmentArm.astype(int)

bot_map = {
    '11' : ('Econ',   'Unrestricted'),
    '12' : ('Econ',   'Diverse'),
    '13' : ('Econ',   'Targeted'),
    '21' : ('ML',     'Unrestricted'),
    '22' : ('ML',     'Diverse'),
    '23' : ('ML',     'Targeted'),
    '31' : ('Random', 'Unrestricted'),
    '32' : ('Random', 'Diverse'),
    '33' : ('Random', 'Targeted')
 }

bot_types = ['11','12','13','21','22','23','31','32','33']
bot_label = []
for x in bot_types: bot_label.append(bot_map[x][0][:2] +': ' + bot_map[x][1][0])

NeuralNet_DataFrame['TreatmentArm_Label'] = 0
for x in bot_types:
    _label  = bot_map[x][0][:2] +': ' + bot_map[x][1][0]
    NeuralNet_DataFrame['TreatmentArm_Label'][NeuralNet_DataFrame['TreatmentArm'] == int(x)] = _label



#=====================================================================================================
#GRAPHS:

#Cummulative interaction by second
ax = sns.lineplot(x="times", y="Cummulative", data=allinteractions, color=gcolor)
for tick in ax.get_xticklabels(): tick.set_rotation(45)

#Frequency of interaction by hour
ax = sns.lineplot(x="hours", y="Interacted", data=allinteractions.groupby('hours').sum().reset_index(), color='salmon')
for tick in ax.get_xticklabels(): tick.set_rotation(45)
plt.show()


ax = sns.lineplot(x="hour", y="Interacted", data=allinteractions[allinteractions['event_name'] == 'initialQuestion'].groupby('hour').sum().reset_index(), color='maroon', label='Initial question')
ax = sns.lineplot(x="hour", y="Interacted", data=allinteractions[allinteractions['event_name'] == 'askInstitution'].groupby('hour').sum().reset_index(), color='red', label='Ask institution')
ax = sns.lineplot(x="hour", y="Interacted", data=allinteractions[allinteractions['event_name'] == 'askCareer'].groupby('hour').sum().reset_index(), color='green', label='Ask program')
ax = sns.lineplot(x="hour", y="Interacted", data=allinteractions[allinteractions['event_name'] == 'askLevel'].groupby('hour').sum().reset_index(), color='salmon', label='Ask level')
ax = sns.lineplot(x="hour", y="Interacted", data=allinteractions[allinteractions['event_name'] == 'OPTIONS'].groupby('hour').sum().reset_index(), color='gray', label='Brain options')
ax = sns.lineplot(x="hour", y="Interacted", data=allinteractions[allinteractions['event_name'] == 'OPTIONS_SELECTION'].groupby('hour').sum().reset_index(), color='orange', label='Select Option')
for tick in ax.get_xticklabels(): tick.set_rotation(45)
plt.show()

last_date = list(set(wageDeviation['days']))
last_date.sort()
last_date = str(last_date[-1])
#Wage diferential by day
fig, ax1 = plt.subplots()
ax = sns.lineplot(x="days", y="interaction", data=wageDeviation[wageDeviation['days']<last_date].groupby(['days']).sum().reset_index(), color='blue')
ax.set_xlabel('')
ax.set_ylabel('Number of interactions', color='tab:blue')
ax.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2 = sns.lineplot(x="days", y="absWageDeviation", data=wageDeviation[wageDeviation['days']<last_date], color='red')
ax2.set_ylabel('Wage Differential', color='tab:red')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='tab:red')
for tick in ax.get_xticklabels(): tick.set_rotation(45)
plt.show()


#Get number of interactions
fig, ax = plt.subplots()
ax = sns.lineplot(x="numberMenu", y="absWageDeviation", data=wageDeviation[wageDeviation['days']<last_date], color=gcolor)
ax.set_xlabel('Number of Menu')
ax.set_ylabel('Percentage deviation from true value', color='tab:blue')
ax.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2 = sns.lineplot(x="numberMenu", y="N_total", data=wageDeviation[wageDeviation['days']<last_date].groupby(['numberMenu']).sum().reset_index(), color='gray')
ax2.set_ylabel('Number of observations', color='tab:red')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='tab:red')
labels = [int(x) for x in ax.get_xticks().tolist()]
labels[-2]='>9'
ax.set_xticklabels(labels)
plt.show()


fig, ax = plt.subplots()
ax = sns.lineplot(x="numberMenu", y="absWageDeviation", hue='Max_interaction',err_style="bars", ci=0, data=wageDeviation[wageDeviation['days']<last_date], color=gcolor)
ax.set_xlabel('Number of Menu')
ax.set_ylabel('Percentage deviation from true value', color='tab:blue')
ax.tick_params(axis='y', labelcolor='tab:blue')
labels = [int(x) for x in ax.get_xticks().tolist()]
labels[-2]='>9'
ax.set_xticklabels(labels)
plt.show()


#Get % of dropout - OutsideOption:

#Number of students assigned to each treatment:
seedAssignment['count'] = 1
ax = sns.barplot(x="bot", y="count", data=seedAssignment.groupby(['bot']).sum().reset_index(), color=gcolor, capsize=.2)
ax.set_xlabel('')
ax.set_ylabel('Number of treatments assigned')
ax.set_xticklabels(bot_label)
plt.show()

#Number of outside option clicks
interactions['outsideOption'] = (interactions['Selection'] == -999)
ax = sns.barplot(x="bot", y="outsideOption", data=interactions.groupby(['bot']).sum().reset_index(), color=gcolor, capsize=.2)
ax.set_xlabel('')
ax.set_ylabel('Number of outside option clicks')
ax.set_xticklabels(bot_label)
plt.show()

#Percentage of outside option selected
interactions['outsideOption'] = (interactions['Selection'] == -999)
ax = sns.barplot(x="bot", y="outsideOption", data=interactions.groupby(['bot']).mean().reset_index(), color=gcolor, capsize=.2)
ax.set_xlabel('')
ax.set_ylabel('Number of outside option clicks')
ax.set_xticklabels(bot_label)
plt.show()

#Number of menus shown by treatment arm:
interactions['count'] = 1
ax = sns.barplot(x="bot", y="count", data=interactions.groupby(['bot']).sum().reset_index(), color=gcolor, capsize=.2)
ax.set_xlabel('')
ax.set_ylabel('Number of menus showed by treatment')
ax.set_xticklabels(bot_label)
plt.show()



#Probability
Probability = {}
for x in bot_types[0:3]:
    _label  = bot_map[x][0] +': ' + bot_map[x][1]
    Probability[_label] = NeuralNet_DataFrame['Probability'][NeuralNet_DataFrame['TreatmentArm'] == int(x)]
    g = sns.kdeplot(Probability[_label], legend = True , label = _label)
plt.show()

#Expected Value
ExpectedValue = {}
for x in bot_types[3:6]:
    _label  = bot_map[x][0] +': ' + bot_map[x][1]
    ExpectedValue[_label] = NeuralNet_DataFrame['ExpectedValue'][NeuralNet_DataFrame['TreatmentArm'] == int(x)]
    g = sns.kdeplot(ExpectedValue[_label], legend = True , label = _label)
plt.show()
