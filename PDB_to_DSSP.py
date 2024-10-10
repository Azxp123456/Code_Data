import argparse
import json
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import numpy as np
import time
import os


def pdb_to_hssp(pdb_file_path, rest_url):
    """
    This sample client takes a PDB file, sends it to a REST service, which
     creates HSSP data. The HSSP data is then output to the console.
    example：
    python pdb_to_hssp.py 1crn.pdb https://www3.cmbi.umcn.nl/xssp/
    """

    files = {'file_': open(pdb_file_path, 'rb')}

    url_create = '{}api/create/pdb_file/hssp_hssp/'.format(rest_url)
    r = requests.post(url_create, files=files)
    r.raise_for_status()

    job_id = json.loads(r.text)['id']
    print("Job submitted successfully. Id is: '{}'".format(job_id))

    ready = False
    while not ready:
        url_status = '{}api/status/pdb_file/hssp_hssp/{}/'.format(rest_url, job_id)

        r = requests.get(url_status)
        r.raise_for_status()

        status = json.loads(r.text)['status']
        print("Job status is: '{}'".format(status))

        if status == 'SUCCESS':
            ready = True
        elif status in ['FAILURE', 'REVOKED']:
            raise Exception(json.loads(r.text)['message'])
        else:
            time.sleep(5)
    else:
        url_result = '{}api/result/pdb_file/hssp_hssp/{}/'.format(rest_url, job_id)

        r = requests.get(url_result)
        r.raise_for_status()
        result = json.loads(r.text)['result']

        return result


def WebDriver():
    """Functional Description：Automatically read PDB files in the folder,
               Automatically open the browser and switch to the pre-set download address,
               Automatically enter the DSSP Web server and automatically process PDB files,
               Automatically modify the processed file name and automatically click to download.
               In short, the process of processing each file is automatically completed.
    """

    folder_path = "\PDB_folder_path"
    file_list = os.listdir(folder_path)

    # Set Chrome options
    chrome_options = Options()
    chrome_options.add_experimental_option('prefs', {
        'download.default_directory': '/path/to/your/directory',
        'download.prompt_for_download': False,
        'download.directory_upgrade': True,
        'safebrowsing.enabled': True
    })

    # Create a WebDriver instance for the Chrome browser
    driver = webdriver.Chrome(options=chrome_options)

    # Open the download settings page
    driver.get("chrome://settings/downloads")

    # Waiting time
    time.sleep(5)

    # You can manually set the default file download address during the waiting time

    # Open the webpage
    driver.get("https://www3.cmbi.umcn.nl/xssp/")

    count = 0
    for file_name in file_list:
        clear_button = driver.find_element(By.ID, "btn_clear")
        clear_button.click()
        time.sleep(2)

        # Select the input box as PDB File and the output as classic DSSP
        input_select = Select(driver.find_element(By.ID, "input_type"))
        input_select.select_by_value("pdb_file")
        output_select = Select(driver.find_element(By.ID, "output_type"))
        output_select.select_by_value("dssp")

        file_path = os.path.join(folder_path, file_name)

        browse_button = driver.find_element(By.ID, "file_")
        browse_button.send_keys(file_path)

        submit_button = driver.find_element(By.XPATH, "//button[@class='btn btn-success']")
        submit_button.click()

        status_changed = False
        wait_timeout = 60

        # Use WebDriverWait to wait for the status variable to change to "SUCCESS"
        wait = WebDriverWait(driver, wait_timeout)
        try:
            wait.until(EC.text_to_be_present_in_element((By.ID, "status"), "SUCCESS"))
            status_changed = True
        except:
            status_changed = False

        time.sleep(3)

        if status_changed:
            download_button = driver.find_element(By.ID, "download")
            current_download_value = download_button.get_attribute("download")
            new_file_name = os.path.splitext(file_name)[0]
            driver.execute_script(f'arguments[0].setAttribute("download", "{new_file_name}.DSSP")', download_button)

            download_button.click()
            print("Download button clicked")
            count += 1

            time.sleep(5)

            # Return to original link
            driver.get("https://www3.cmbi.umcn.nl/xssp/")

            time.sleep(5)

    print(count)


def get_file_names(folder_path):
    file_names = set()
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_name, _ = os.path.splitext(file)
            file_names.add(file_name)
    return file_names


if __name__ == "__main__":
    WebDriver()


