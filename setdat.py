import random
import pandas as pd
import datetime

th_typ= ["Port Scan", "Malware","Unusual Traffic","Normal Traffic", "Unauthorized Login", "Brute Force", "Trojans", "DDoS"]
sev_typ= ["Low", "Intermediate", "High"]
stat = ["Investigating","Active", "Resolved"]

log=[]

for i in range(2000):
    timestmp = datetime.datetime.now().strftime("%m/%d/%Y %I:%M:%S %p") 
    log_typ = random.choice(th_typ)
    severity = random.choice(sev_typ)
    status = random.choice(stat)

    if log_typ != "Normal Traffic":
        anomaly = 1
    else:
        anomaly = 0

    log.append([timestmp, log_typ, severity, status, anomaly])


logs_data = pd.DataFrame(log, columns=["Timestamp", "Threat Type" ,"Severity", "Status", "Anomaly"])
logs_data.to_csv("log_data.csv", index = False)
