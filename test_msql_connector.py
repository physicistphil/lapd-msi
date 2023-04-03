import mysql.connector
import re
import numpy as np

num_pat = re.compile(r"\b\d+")
run_pat = re.compile(r"\w+\.h5")

for i in range(10):
    cnx = mysql.connector.connect(user='readonly', password='uLTentENEnTIncioUtIO',
                              host='192.168.7.3')
    curr_datarun = "datarun 176{} ICRF_campaign".format(i)

    num = int(num_pat.search(curr_datarun + ".h5").__getitem__(0))
    run = run_pat.search(curr_datarun + ".h5").__getitem__(0)[:-3]
    cursor = cnx.cursor(buffered=True)
    try:
        cursor.execute("SELECT data_run_name FROM {}.data_runs WHERE data_run_id in ({});".format(run, num))
        (run_name,) = cursor.fetchall()[0]
    except Exception as e:
        print(e)
    cursor.close()
    print(num)
    print(run)
    print(run_name)

    cnx.disconnect()
