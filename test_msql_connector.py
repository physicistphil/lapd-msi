import mysql.connector
import re
import numpy as np

cnx = mysql.connector.connect(user='readonly', password='uLTentENEnTIncioUtIO',
                              host='192.168.7.3')
num_pat = re.compile(r"\b\d+")
run_pat = re.compile(r"\w+\.h5")

curr_datarun = "datarun 1766 ICRF_campaign"

num = int(num_pat.match(curr_datarun + ".h5").__getitem__(0))
run = run_pat.search(curr_datarun + ".h5").__getitem__(0)[:-3]

cursor = cnx.cursor(buffered=True)
try:
    # Want to select just the single datarun here...
    cursor.execute("SELECT data_run_name FROM {}.data_runs WHERE datarun_id in ({});".format(run, num))
    file_mapping = cursor.fetchall()
except Exception as e:
    print(e)
cursor.close()

cnx.disconnect()

print(file_mapping)
print(file_mapping[0])
print(file_mapping[1])
