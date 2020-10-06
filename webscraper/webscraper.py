import urllib.request
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
from http.client import RemoteDisconnected 
import requests
from bs4 import BeautifulSoup as soup
import os
import time

baseurl = 'https://www.autoblog.com'
carurl = 'https://www.autoblog.com/research/?find=makes'
headers = {'User-Agent': 'Mozilla/5.0'}

def getListMakes():
    """Haalt een list met alle merknamen en hun bijhorende links op"""

    carlist = []

    try:
        pagehtml = requests.get(carurl, headers=headers)  #get the page
        souppage = soup(pagehtml.text, "html.parser")    #html parsing

        ulcars = souppage.find("ul", {"class":"items columnize"})
        licars = ulcars.find_all("li")

        for item in licars:
            a = item.find("a")
            make = a.contents[0]
            link = a.get('href')

            carlist.append({"make": make, "link": link})
    except RemoteDisconnected:
        print("!!!!!!!!!!!!!!!!!!!!RemoteDisconnected!!!!!!!!!!!!!!!!!!!!")
        pass
    except urllib.request.HTTPError:
        print("*/*/*/*/*/*/*/*/*/*/*/Error/*/*/*/*/*/*/*/*/*/*/*")
        
    return carlist

def getModels(merk):
    """Volgt de link van een belpaald merk naar de pagina van het merk
    om daar het model te kiezen en geeft een lijst van de verschillende
    modellen terug"""

    modellist = []

    try:
        pagehtml = requests.get(baseurl + merk["link"], headers=headers)  #get the page      
        souppage = soup(pagehtml.text, "html.parser")

        titel = souppage.find("h1").contents[0]  #kunnen we gebruiken voor test
        if(titel != merk["make"]):
            exit(0)
        
        div = souppage.find("div", {"class":"dropdown"})
        optiongroup = div.find("optgroup")
        options = optiongroup.find_all("option")

        for item in options:
            model = item.contents[0]
            link = item.get("value")
            modellist.append({"model": model, "link": link})
    except RemoteDisconnected:
        print("!!!!!!!!!!!!!!!!!!!!RemoteDisconnected!!!!!!!!!!!!!!!!!!!!")
        pass
    except urllib.request.HTTPError:
        print("*/*/*/*/*/*/*/*/*/*/*/Error/*/*/*/*/*/*/*/*/*/*/*")
    
    return modellist

def getYears(model):
    """Volgt de link naar model pagina en geeft een lijst met alle jaartallen 
    terug en hun links"""

    yearlist = []

    try:
        pagehtml = requests.get(baseurl + model["link"], headers=headers)  #get the page
        souppage = soup(pagehtml.text, "html.parser")

        title = souppage.find("h1").contents[1].strip()

        if(title != model["model"]):
            exit(0)

        div = souppage.find("div", {"class":"dropdown"})
        if div is not None:
            options = div.find_all("option")
            del options[0]

            for item in options:
                year = item.contents[0]
                link = item.get("value")
                yearlist.append({"year": year, "link": link})
    except RemoteDisconnected:
        print("!!!!!!!!!!!!!!!!!!!!RemoteDisconnected!!!!!!!!!!!!!!!!!!!!")
        pass
    except urllib.request.HTTPError:
        print("*/*/*/*/*/*/*/*/*/*/*/Error/*/*/*/*/*/*/*/*/*/*/*")

    return yearlist

def getLinkToPhotos(year):
    """Haalt de link op om naar alle foto's van de auto te gaan"""
    link = "skip"
    try:
        pagehtml = requests.get(baseurl + year["link"], headers=headers)  #get the page
        souppage = soup(pagehtml.text, "html.parser")

        div = souppage.find("div", {"class":"slider masthead-photo"})
        if div is not None:
            a = div.find("a")
            link = a.get("href")
    except RemoteDisconnected:
        print("!!!!!!!!!!!!!!!!!!!!RemoteDisconnected!!!!!!!!!!!!!!!!!!!!")
        pass
    except urllib.request.HTTPError:
        print("*/*/*/*/*/*/*/*/*/*/*/Error/*/*/*/*/*/*/*/*/*/*/*")

    return link

def downloadPhotos(link, make, model, year):
    """Download alle foto's van het model en jaar en steekt deze in een juiste map"""
    try:
        pagehtml = requests.get(baseurl + link, headers=headers)  #get the page
        souppage = soup(pagehtml.text, "html.parser")

        divphotos = souppage.find_all("div", {"class":"rsContent"})

        print(make + "\t" + model + "\t" + year +"\t START")
        for item in divphotos:
            
            a = item.find("a")
            photolink = a.get("href")
            if ("21005" in photolink) or ("109.jpg" in photolink):
                print("-----------------------------------------FRONTVIEW-----------------------------------------")
                urllib.request.urlretrieve(photolink, "data/frontview/" + make+"_"+model+"_"+year+"_"+"FRONT"+".jpg")
                
            elif ("21006" in photolink) or ("113.jpg" in photolink):
                print("-----------------------------------------BACKVIEW------------------------------------------")
                urllib.request.urlretrieve(photolink, "data/backview/" + make+"_"+model+"_"+year+"_"+"BACK"+".jpg")
            
            elif ("21003" in photolink) or ("112.jpg" in photolink):
                print("-----------------------------------------SIDEVIEW------------------------------------------")
                urllib.request.urlretrieve(photolink, "data/sideview/" + make+"_"+model+"_"+year+"_"+"SIDE"+".jpg")
            elif ("21001" in photolink) or ("101.jpg" in photolink):
                print("--------------------------------------FRONTSIDEVIEW----------------------------------------")
                urllib.request.urlretrieve(photolink, "data/frontsideview/" + make+"_"+model+"_"+year+"_"+"FRONTSIDE"+".jpg")
            elif ("21002" in photolink) or ("102.jpg" in photolink):
                print("--------------------------------------BACKSIDEVIEW-----------------------------------------")
                urllib.request.urlretrieve(photolink, "data/backsideview/" + make+"_"+model+"_"+year+"_"+"BACKSIDE"+".jpg")
        print(make + "\t" + model + "\t" + year +"\t DONE")

    except RemoteDisconnected:
        print("!!!!!!!!!!!!!!!!!!!!RemoteDisconnected!!!!!!!!!!!!!!!!!!!!")
        pass
    except urllib.request.HTTPError:
        print("*/*/*/*/*/*/*/*/*/*/*/Error/*/*/*/*/*/*/*/*/*/*/*")

def main():
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/frontview"):
        os.mkdir("data/frontview")
    if not os.path.exists("data/backview"):
        os.mkdir("data/backview")
    if not os.path.exists("data/sideview"):
        os.mkdir("data/sideview")
    if not os.path.exists("data/frontsideview"):
        os.mkdir("data/frontsideview")
    if not os.path.exists("data/backsideview"):
        os.mkdir("data/backsideview")

    carlist = getListMakes()
    for car in carlist:
        modellist = getModels(car)
        for model in modellist:
            yearlist = getYears(model)
            for year in yearlist:
                photolink = getLinkToPhotos(year)
                if photolink != "skip":
                    downloadPhotos(photolink, car["make"], model["model"], year["year"])
            print("Wachten...")
            time.sleep(37)
            

main()