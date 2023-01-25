from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import requests
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import hashlib

from tqdm import tqdm
from time import sleep

import pandas as pd

curdir = os.path.dirname(os.path.realpath(__file__))

options = webdriver.ChromeOptions() 
options.add_argument("--lang=en")
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

driver.get("https://www.youtube.com")
sleep(0.1)
driver.find_element(By.XPATH, "//*[@id=\"content\"]/div[2]/div[6]/div[1]/ytd-button-renderer[1]/yt-button-shape/button/yt-touch-feedback-shape/div/div[2]").click()

def find_size(lrow, lcol):
    # look for size
    val =  0
    for row in range(1, lrow):
        for col in range(1, lcol):
            try:
                driver.find_element(By.XPATH, f"/html/body/ytd-app/div[1]/ytd-page-manager/ytd-browse[1]/ytd-two-column-browse-results-renderer/div[1]/ytd-rich-grid-renderer/div[6]/ytd-rich-grid-row[{row}]/div/ytd-rich-item-renderer[{col}]/div/ytd-rich-grid-media/div[1]/div[2]/div[1]/h3/a/yt-formatted-string")
                val +=1
            except:
                break
    return val

cols = find_size(2, 100) + 1
rows = find_size(100, 2) + 1

print("[TUBE DOWNLOADER] found col size: {} and row size: {}".format(cols, rows)) 

limit = int(input("[TUBE DOWNLOADER] how many images should i donwload? "))
count = 0

df = None
if os.path.exists(os.path.join(curdir, "data", "labels", "labels.csv")):
    df = pd.read_csv(os.path.join(curdir, "data", "labels", "labels.csv"), sep=",")
else:
    df = pd.DataFrame(columns=["image", "title", "views"])

while count < limit:
    driver.refresh()
    for row in range(1, rows):
        for col in range(1, cols):
            src = None
            title = None
            views = None
            try:
                img = driver.find_element(By.XPATH, f"/html/body/ytd-app/div[1]/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-rich-grid-renderer/div[6]/ytd-rich-grid-row[{row}]/div/ytd-rich-item-renderer[{col}]/div/ytd-rich-grid-media/div[1]/ytd-thumbnail/a/yt-image/img")
                src = img.get_attribute('src')
            except:
                print("[TUBE DOWNLOADER] Image not found")
                continue

            if src != "" and src is not None:
                try:
                    title = driver.find_element(By.XPATH, f"/html/body/ytd-app/div[1]/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-rich-grid-renderer/div[6]/ytd-rich-grid-row[{row}]/div/ytd-rich-item-renderer[{col}]/div/ytd-rich-grid-media/div[1]/div[2]/div[1]/h3/a/yt-formatted-string")
                    title = title.text
                except:
                    print("[TUBE DOWNLOADER] title not found")
                    continue
                
                try:
                    views = driver.find_element(By.XPATH, f"/html/body/ytd-app/div[1]/ytd-page-manager/ytd-browse/ytd-two-column-browse-results-renderer/div[1]/ytd-rich-grid-renderer/div[6]/ytd-rich-grid-row[{row}]/div/ytd-rich-item-renderer[{col}]/div/ytd-rich-grid-media/div[1]/div[2]/div[1]/ytd-video-meta-block/div[1]/div[2]/span[1]")
                    views = views.text
                except:
                    print("[TUBE DOWNLOADER] views not found")
                    continue


                if views.split(" ")[0].endswith("K"):
                    views = float(views.replace(",", ".").replace("K", "").split(" ")[0]) * 1000
                elif views.split(" ")[1] == "Mln" or views.split(" ")[0].endswith("M"):
                    views = float(views.replace(",", ".").replace("M", "").split(" ")[0]) * 1000000
                elif views.split(" ")[1] == "Mrd" or views.split(" ")[1] == "Bln" or views.split(" ")[0].endswith("B"):
                    views = float(views.replace(",", ".").replace("B", "").split(" ")[0]) * 1000000000
                else:
                    views = float(views.replace(".", "").split(" ")[0])
                    
                filename = hashlib.md5(title.encode('utf-8')).hexdigest()
                img_data = requests.get(src).content

                if img_data is not None and img_data != b"":
                    filename = '{}.jpg'.format(filename)
                    if not df['image'].str.contains(filename).any():
                        with open(os.path.join(curdir, "data", "images", filename), 'wb') as handler:
                                handler.write(img_data)
                                
                                print("[TUBE DOWNLOADER] TITLE: {} - VIEWS: {} - ITERATION: {}".format(title, views, count))
                                new_row = pd.DataFrame({"image": [filename], "title": [title], "views": [views]})
                                df = pd.concat([df, new_row], ignore_index=True, sort=False)
                                df.to_csv(os.path.join(curdir, "data", "labels", "labels.csv"), index=False, sep=",")
                                count += 1
                else:
                    print("TUBE DOWNLOADER: error, src not found")

driver.quit()