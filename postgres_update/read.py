import json
import os
import time, json
import pandas as pd
import psycopg2
from redis_om import get_redis_connection


def update_table(table="predictions"):
    conn = None

    # connect to the PostgreSQL server
    conn = psycopg2.connect(
        database="postgres", user="postgres", password="postgres", host="192.168.1.136", port="5834"
    )
    cursor = conn.cursor()
    postgreSQL_select_Query = "select count(*) from predictions limit 10"
    cursor.execute(postgreSQL_select_Query)
    print("Opened database successfully")
    print("Selecting rows from mobile table using cursor.fetchall")
    mobile_records = cursor.fetchall()
    print(mobile_records)
    cursor.close()


update_table()