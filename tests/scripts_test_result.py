import subprocess
import pandas as pd
import re
import os
import sys


def run_tox(env):
    log_file_path = "./tox.log"
    try:
        with open(log_file_path, "w") as log_file:
            if env is None:
                tox = ["tox"]
            else:
                tox = ["tox", "-e", env]
            subprocess.run(tox, stdout=log_file, stderr=subprocess.STDOUT)
            print("Tox tests ran successfully.")
    except subprocess.CalledProcessError as e:
        print("Tox tests failed.")
        print(e.output.decode())


def parse_tox_log(log_path):
    results = {}
    try:
        with open(log_path, "r") as file:
            for line in file:
                if "PASSED" in line:
                    func = re.search(r"/([^/]*?)::", line)
                    results[func.group(1)] = "PASSED"
                if "FAILED" in line:
                    func = re.search(r"/([^/]*?)::", line)
                    results[func.group(1)] = "FAILED"
    except FileNotFoundError:
        print("Log file not found.")
    return pd.DataFrame(list(results.items()), columns=["Test File", "Result"])


def create_markdown_table(df):
    return df.to_markdown(index=False)


if __name__ == "__main__":
    env = None
    if len(sys.argv) > 1:
        env = sys.argv[1]
    print(f"tox run with {env} environment")
    run_tox(env)
    df = parse_tox_log("./tox.log")
    markdown_output = create_markdown_table(df)
    print(markdown_output)
    os.remove("./tox.log")
