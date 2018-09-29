import pandas as pd
import numpy as np
import scipy, random, copy, h5py, pandas, math, csv, sys, os, pickle, re, json
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn import preprocessing
from numpy.random import RandomState
from scipy import ndimage
from scipy import stats
from scipy import ndimage
path = os.getcwd()

sub_routines = os.path.join(path,'..','QueVasaEstudiar','python', 'sub_routines')
sys.path.append(sub_routines)
from neural_predictor import neural_predictor
from Import_MenuParameters import Import_MenuParameters

#1. Import data and parameters:
root = "C:/Users/Franco/GitHub" #Franco's MSI

csv_path = root + "/Bots_Colombia/data/"

def loadOutcomes():
    outcomes = {}
#    with open('python/Options_Outcomes_11_29_2017.csv') as csvfile:
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

#1. Set up and import data:
#==========================

#1.1. Import Data:
HEOptions = arange_outcomes(outcomes)

with open(root+'/QueVasaEstudiar/interactions.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(";") for line in stripped if line)
    with open(root+'/QueVasaEstudiar/log.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

allinteractions = pd.read_csv(root+'/QueVasaEstudiar/log.csv', error_bad_lines=False)
allinteractions = allinteractions[(allinteractions.user != 'default-user') & (allinteractions.user.str.len() == 20)].reset_index(drop=True)

allinteractions["year"] = allinteractions.timestamp.str[0:4].astype(int)
allinteractions["month"] = allinteractions.timestamp.str[5:7].astype(int)
allinteractions["day"] = allinteractions.timestamp.str[8:10].astype(int)
allinteractions["hour"] = allinteractions.timestamp.str[11:13].astype(int)
allinteractions["minute"] = allinteractions.timestamp.str[14:16].astype(int)
allinteractions["second"] = allinteractions.timestamp.str[17:19].astype(int)

DF1 = pd.read_csv(root+'/QueVasaEstudiar/Student_Features_Fall2018.csv'.format(100))
DF2 = pd.read_csv(root+'/Bots_Colombia/ModelEstimation/deeplearning_estimation/interactions_bot.csv'.format(100))
interactions = DF1.merge(DF2, left_on = 'Student_ID', right_on = 'Student_ID', how = 'left')
interactions = interactions.merge(Student_Features, left_on = 'Student_ID', right_on = 'Student_ID')
interactions = interactions.merge(Student_Features, left_on = 'Student_ID', right_on = 'Student_ID')


#=====================================================================================================


after18 = data[(data.year == 2018) & (data.month >= 9) & (data.day >= 18)]

after18=after18[(after18.user != "992") & (after18.user != "YBOF9JBM8DE9ME6RE1K4") & (after18.user != "EMF393CY0PX7ESPW0BNQ")]
total_interact = len(after18)
unique_users = after18.user

N_uu = len(unique_users.drop_duplicates(keep='first', inplace=False))

summaryafter18 = after18["user"].value_counts()
summaryafter18 = summaryafter18.to_frame(name=None)
summaryafter18.columns = ['interactions']
summaryafter18['url_id'] = summaryafter18.index
#summaryafter18.loc[summaryafter18['interactions'] > 100, 'interactions'] = 100

summaryafter18.mean()

students_data = pd.read_csv(interactions_path + "/Student_Features_Fall2018.csv")
students_data.set_index('url_id', inplace=True)
result = pd.concat([summaryafter18, students_data], axis=1, join_axes=[summaryafter18.index])


day_set = (students_data.day_group == "D01") | (students_data.day_group == "D02") |(students_data.day_group == "D03")
engagement_rate= round(100*N_uu/len(students_data[day_set].drop_duplicates(keep='first', inplace=False)),2)
engagement_rate_total = round(100*N_uu/len(students_data.drop_duplicates(keep='first', inplace=False)),2)

print("Estadística generales: ")
print("Interacciones: ",N_uu,"/nTasa de Engagement: ",engagement_rate,"%", "/nPromedio de interacciones por usuario: ",round(summaryafter18.mean()[0],2),sep = "")
print("/nDesglose por grupo diario: ")
print(result["day_group"].value_counts())
print("/nDesglose por ubicación: ")
print(result["Student_Location_Name"].value_counts())

#Type of interaction:
A = allinteractions.event_name.value_counts()
askAboutProgram   = A[0]
initialQuestion   = A[1]
askInstitution    = A[2]
truth             = A[3]
askCareer         = A[4]
OPTIONS           = A[5]
askArea           = A[6]
OPTIONS_SELECTION = A[7]
askLevel          = A[8]

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

ax = sns.lineplot(x="day", y="second",hue="event", data=allinteractions)

#Change report in spreadsheets:
#=============================

from oauth2client.service_account import ServiceAccountCredentials
import gspread

json_path="C:/Users/Gonzalo/Dropbox (JPAL LAC)/ConsiliumBots/Colombia/ReportPanel/My Project-5584097d5fa1.json"
dash_key = '1QpXG8KeEYzsMam2SSZE8ezu4yoYHYaj1oWNe9kkHtQ8'
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = (ServiceAccountCredentials.from_json_keyfile_name(json_path, scope))
client = gspread.authorize(creds)
db_ws = client.open_by_key(dash_key).worksheet('Dashboard')

db_ws.update_cell(1,2, total_interact)
db_ws.update_cell(2,2, N_uu)
db_ws.update_cell(3,2, engagement_rate)
db_ws.update_cell(4,2, round(summaryafter18.mean()[0],2))
result.interactions.median()
