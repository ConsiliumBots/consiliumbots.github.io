import os
import csv
import json
import csv
import re
import sys
import pandas as pd

def getBaseDirectory():
    try:
        return os.path.dirname(__file__)
    except NameError:
        return ''

interactions_path  = "C:/Users/Franco/GitHub/QueVasaEstudiar"

with open(interactions_path + '/interactions1.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(";") for line in stripped if line)
    with open(interactions_path + '/log.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

data = pd.read_csv(interactions_path + '/log.csv', error_bad_lines=False)
largo=len(data)


data["year"] = data.timestamp.str[0:4]
data["month"] = data.timestamp.str[5:7]
data["day"] = data.timestamp.str[8:10]
data["hour"] = data.timestamp.str[11:13]
data["minute"] = data.timestamp.str[14:16]
data["second"] = data.timestamp.str[17:19]

data["month"]=pd.to_numeric(data.month)
data["year"]=pd.to_numeric(data.year)
data["day"]=pd.to_numeric(data.day)

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
