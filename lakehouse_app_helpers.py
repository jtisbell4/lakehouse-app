import json
import re  # for nice rendering
import time

import numpy as np  # for nice rendering
import pandas as pd  # for nice rendering
import requests

host = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiUrl()
    .getOrElse(None)
)
token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .getOrElse(None)
)
auth = {"Authorization": "Bearer " + token}


def list():
    json = requests.get(f"{host}/api/2.0/preview/apps", headers=auth).json()

    # now render it nicely
    df = pd.DataFrame.from_dict(json["apps"], orient="columns")
    df["state"] = df["status"].apply(lambda x: x["state"])
    df["status message"] = df["status"].apply(lambda x: x["message"])
    df.drop("status", axis=1, inplace=True)
    df = df[["name", "state", "status message", "create_time", "url"]]
    df["logz"] = df["url"].apply(lambda x: "" if x == "" else x + "/logz")
    html = df.to_html(index=False)
    html = re.sub(
        r"https://((\w|-|\.)+)\.databricksapps\.com/logz",
        r'<a href="https://\1.databricksapps.com/logz">Logz</a>',
        html,
    )
    html = re.sub(
        r"(<td>)(https://((\w|-|\.)+)\.databricksapps\.com)",
        r'\1<a href="\2">Link</a>',
        html,
    )
    html = re.sub(
        r"(<td>)(ERROR)(</td>)", r'\1<span style="color:red">\2</span>\3', html
    )
    html = re.sub(
        r"(<td>)(RUNNING)(</td>)", r'\1<span style="color:green">\2</span>\3', html
    )
    html = (
        "<style>.dataframe tbody td { text-align: left; font-size: 14 } .dataframe th { text-align: left; font-size: 14 }</style>"
        + html
    )
    displayHTML(html)


def create(app_name, app_description="This app does something"):
    requests.post(
        f"{host}/api/2.0/preview/apps",
        headers=auth,
        json={"name": app_name, "spec": {"description": app_description}},
    ).json()

    # The create API is async, so we need to poll until it's ready.
    for _ in range(10):
        time.sleep(5)
        response = requests.get(
            f"{host}/api/2.0/preview/apps/{app_name}", headers=auth
        ).json()
        if response["status"]["state"] != "CREATING":
            break
    response


def deploy(app_name, source_code_path):
    # Deploy starts the pod, downloads the source code, install necessary dependencies, and starts the app.
    response = requests.post(
        f"{host}/api/2.0/preview/apps/{app_name}/deployments",
        headers=auth,
        json={"source_code_path": source_code_path},
    ).json()
    deployment_id = response["deployment_id"]

    # wait until app is deployed. We still do not get the real app state from the pod, so even though it will say it is done, it may not be.
    # Especially the first time you deploy. We're working on not restarting the pod on the second deploy.
    # Logs: if you want to see the app logs, go to {app-url}/logz.
    for _ in range(10):
        time.sleep(5)
        response = requests.get(
            f"{host}/api/2.0/preview/apps/{app_name}/deployments/{deployment_id}",
            headers=auth,
        ).json()
        if response["status"]["state"] != "IN_PROGRESS":
            break
    response


def details(app_name):
    url = host + f"/api/2.0/preview/apps/{app_name}"
    json = requests.get(url, headers=auth).json()

    # now render it nicely
    df = pd.DataFrame.from_dict(json, orient="index")
    html = df.to_html(header=False)
    html = re.sub(
        r"(<td>)(https://((\w|-|\.)+)\.databricksapps\.com)",
        r'\1<a href="\2">\2</a>',
        html,
    )
    html = (
        "<style>.dataframe tbody td { text-align: left; font-size: 14 } .dataframe th { text-align: left; font-size: 14 }</style>"
        + html
    )
    displayHTML(html)


def delete(app_name):
    url = host + f"/api/2.0/preview/apps/{app_name}"
    json = requests.delete(url, headers=auth).json()
    json


# COMMAND ----------
