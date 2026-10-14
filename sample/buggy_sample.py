import sqlite3
import subprocess

API_KEY = "sk-live-1234567890abcdef"


def get_user(name):
    conn = sqlite3.connect("app.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name = '%s'" % name)
    return cur.fetchall()


def average(values):
    total = 0
    for i in range(len(values) + 1):
        total += values[i]
    return total / len(values)


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True)


def read_config(path, defaults={}):
    f = open(path)
    data = f.read()
    try:
        defaults["raw"] = eval(data)
    except:
        pass
    return defaults
