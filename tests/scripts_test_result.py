import subprocess
import pandas as pd
import re
import os
import sys

"""
The `test-script.R` script is designed to leverage the `tox` command for running
tests across various functions. It will automatically generate a markdown table
to record all the test results. The markdown table will be printed on the terminal
and add to the README.rst

"""


def run_tox(env):
    log_file_path = "./tox.log"
    try:
        with open(log_file_path, "w", encoding="utf-8") as log_file:
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
        with open(log_path, encoding="utf-8") as file:
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


def add_markdown_table_to_readme(markdown_output):
    try:
        with open("./README.rst", encoding="utf-8") as f:
            readme_content = f.read()

        pattern = "----------------------------\n"
        if pattern in readme_content:
            index = readme_content.index(pattern)
            readme_content = readme_content[:index]

        with open("./README.rst", "w", encoding="utf-8") as f:
            f.writelines(readme_content)
            f.write(pattern)
            f.write(markdown_output)
            f.write("\n")
            f.write(pattern)

    except FileNotFoundError:
        print("README.rst not found.")


if __name__ == "__main__":
    env = None
    if len(sys.argv) > 1:
        env = sys.argv[1]
    print(f"tox run with {env} environment")
    run_tox(env)
    df = parse_tox_log("./tox.log")
    markdown_output = create_markdown_table(df)
    print(markdown_output)
    add_markdown_table_to_readme(markdown_output)
    os.remove("./tox.log")
