# coding: utf-8

# In[4]:

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen

import os
import re
import sys
import time
import random

link="https://www.google.com/"
#driver = webdriver.Chrome(ChromeDriverManager().install())  # Path of "chromdriver.exe"
driver= webdriver.Chrome(r'C:\Users\Lee Wook Jin\PycharmProjects\instgram_crawling_kangs\chromedriver\chromedriver.exe')
driver.get(link)

# In[46]:

def SearchPosting(Search, max_iteration=100, want_post=2000):
    # Search : Target name of Search, Ex."love"
    # max_iteration : 33 posts per max_iteration, max_iteration하나당 33개의 게시물을 저장함
    global driver

    url = "https://www.instagram.com/explore/tags/" + str(Search) + "/"
    #    driver = webdriver.Chrome(r"C:\Users\imjg0\Desktop\chromedriver") # Path of "chromdriver.exe"
    driver.get(url)
    time.sleep(3)  # Waiting for Open Web Site

    # Save Post
    reallink = []
    reallink_set={}
    memory=1e8
    iteration = 0
    #while iteration < max_iteration:
    while len(reallink_set)< want_post:
        if len(reallink_set)!=memory:
            memory=len(reallink_set)
            print(memory)
        PageString = driver.page_source
        BSobj = BeautifulSoup(PageString, "lxml")
        for link1 in BSobj.find_all(name="div", attrs={"class": "Nnq7C weEfm"}):
            try:
                # First Post
                title = link1.select('a')[0]
                real = title.attrs['href']
                real = "https://www.instagram.com" + real
                reallink.append(real)
            except:
                break

            try:
                # Second Post
                title = link1.select('a')[1]
                real = title.attrs['href']
                real = "https://www.instagram.com" + real
                reallink.append(real)
            except:
                break

            try:
                # Third Post
                title = link1.select('a')[2]
                real = title.attrs['href']
                real = "https://www.instagram.com" + real
                reallink.append(real)
            except:
                break
            reallink_set=set(reallink)

        iteration += 1
        last_height = driver.execute_script("return document.body.scrollHeight")  # Down Scrolling
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")  # Down Scrolling
        new_height = driver.execute_script("return document.body.scrollHeight")  # Down Scrolling

    return reallink


def SaveImageText(url, Search):
    # url="https://www.instagram.com/p/B8BBJCzAR8g/"
    global driver
    global Search_number

    driver.get(url)
    driver.implicitly_wait(5)
    driver2=driver
    # Image Set in a post
    temp = []
    number_image = 1
    while (1):
        time.sleep(2)
        pageString = driver.page_source
        soup = BeautifulSoup(pageString, "lxml")
        if len(soup.select('img')) == 1:
            break
        try:
            imgs = soup.select('img')[number_image]

        except IndexError:
            break
        #        imgs=soup.select('img')[number_image]
        if 'src' not in imgs.attrs:
            continue
        imgs = imgs.attrs['src']
        temp.append(imgs)
        try:
            driver.find_element_by_class_name("coreSpriteRightChevron").click()  # The number of Image
            number_image += 1

        except NoSuchElementException:
            break
    ##################################### location #####################################
    flag=True
    for i in soup.find_all('script', type="text/javascript"):
        if i.text.split()==[]:
            continue
        else:
            if i.text.split()[0]=='window._sharedData':
                try:
                    data_loc_cap=json.loads(i.text.split('= ')[1].replace(';', ''))
                except:
                    flag=False
                    continue
                data_loc_cap = data_loc_cap['entry_data']['PostPage'][0]
                if 'graphql' not in data_loc_cap:
                    print('search other')
                    continue
                data_loc_cap=data_loc_cap['graphql']['shortcode_media']
                caption = data_loc_cap['edge_media_to_caption']['edges']
                if len(caption) == 0:
                    flag = False
                    continue
                caption = caption[0]['node']['text']
                location = data_loc_cap['location']
                if location==None:
                    flag=False
                    continue
                location=location['name']
                if CorrectTest(location)==False:
                    flag=False
                    continue
            elif i.text.split('(')[0]=='window.__additionalDataLoaded':
                try:
                    dic=json.loads(i.text[i.text.find(',')+1:].replace(');', ''))
                except:
                    flag=False
                    continue
                data_loc_cap=dic['graphql']['shortcode_media']
                caption = data_loc_cap['edge_media_to_caption']['edges']
                if len(caption)==0:
                    flag=False
                    continue
                caption=caption[0]['node']['text']
                location = data_loc_cap['location']
                if location == None:
                    flag = False
                    continue
                location = location['name']
                if CorrectTest(location) == False:
                    flag = False
                    continue
    ###################################################################################

    String = str(soup.select('title')[0])
    if "페이지를 찾을 수 없습니다" in String:  # 페이지가 제거된 경우 넘긴다
        return None
    String = instagram_str(String)  # #tag, @tag의 경우 제거
    String = clean_str(String)  # 데이터 정제
    String = remove_emojis(String)  # 이모티콘 제거

    if not len(String.split()) == 0:
        if (CorrectTest(String) == True) & (flag==True):
            #            print("URL:",url)             #필요시 print
            #            print("English?: ",String)    #필요시 print
            # Saver text
            with open('./img/' + str(Search) + '/' + str(Search) + ".txt", 'a', encoding='utf-8') as tx:
                String = "Post_" + str(Search_number) + '\t' + String + '\n'
                tx.write(String)

            # Saver location
            with open('./img/' + str(Search) + '/' + str(Search) + "_loc.txt", 'a', encoding='utf-8') as tx2:
                String_loc = "Post_" + str(Search_number) + '\t' + location + '\n'
                tx2.write(String_loc)

            # Saver total caption
            with open('./img/' + str(Search) + '/' + str(Search) + "_cap.txt", 'a', encoding='utf-8') as tx3:
                String_cap = "Post_" + str(Search_number) + '\t' + caption + '\n'
                tx3.write(String_cap)

            # Saver image
            num = 1
            for imgUrl in temp:
                try:
                    with urlopen(imgUrl) as f:
                        with open('./img/' + str(Search) + "/" + str(Search) + "_Post" + str(Search_number) + "(" + str(
                                num) + ').jpg', 'wb') as h:
                            img = f.read()
                            h.write(img)
                            num += 1
                except ValueError:
                    break

            Search_number += 1


def instagram_str(string):
    try:
        try:
            string = string.split('님:')[1:][0]
        except IndexError:
            string = "잘못된 언어"

        string = re.sub("#[a-zA-Z0-9\_\-.,!?\'\`]+", "", string)
        string = re.sub("@[a-zA-Z]+", "", string)
        string = re.sub("#", "", string)
        string = re.sub("</title>", "", string)
        string = re.sub("\“", "", string)
        string = re.sub("\”", "", string)
        string = re.sub("\’", " \'", string)
        string = re.sub("…", "", string)
        string = re.sub("\|", "", string)
        string = re.sub(":", " ", string)
        string = re.sub("•", " ", string)
    except KeyboardInterrupt:
        sys.exit()

    return string


def clean_str(string):
    # Yoon kim English Preprocessing Revise
    string = re.sub(r"[.,!?\'\`]", " ", string)
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\’ve", " \'ve", string)
    string = re.sub(r"n\’t", " n\'t", string)
    string = re.sub(r"\’re", " \'re", string)
    string = re.sub(r"\’d", " \'d", string)
    string = re.sub(r"\’ll", " \'ll", string)
    string = re.sub(r"\.", " .", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def remove_emojis(String):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', String)


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)


# emoji_pattern.sub(r'', text)

def CorrectTest(String):
    Before_len = len(re.sub("[^A-Za-z0-9\_\-.,!?\'\`]", "", String))
    After_len = len("".join(String.split()))
    #    print(re.sub("[^A-Za-z0-9\_\-.,!?\'\`]", "", String))
    #    print("".join(String.split()))
    return Before_len == After_len


def CrawlingInstagram(Search, numbers=8, max_iteration=100):
    global Search_number
    global driver

    urles = SearchPosting(Search, max_iteration=max_iteration)
    random.shuffle(urles)
    urls = list(set(urles))
    print("Keyword :", Search, " - ", len(urls), "/", len(urles))

    # Make Folder
    try:
        os.makedirs('./img/' + str(Search))
    except FileExistsError:
        pass

    Search_number = 1
    n = 0
    while Search_number <= numbers:
        print(n,' - ' ,Search_number)
        #if (n==50) & (Search_number==1):
        #    break
        if n>len(urls)-1:
            break
        SaveImageText(urls[n], Search)
        n += 1
        # try:
        #     SaveImageText(urls[n], Search)
        #     n += 1
        # except:
        #     break


def CrawlingInstagramList(SearchSet, numbers=8, max_iteration=100):
    global driver

    driver = webdriver.Chrome(r'C:\Users\Lee Wook Jin\PycharmProjects\instgram_crawling_kangs\chromedriver\chromedriver.exe')  # Path of "chromdriver.exe"
    for Search in SearchSet:
        CrawlingInstagram(Search, numbers=numbers, max_iteration=max_iteration)




import json

with open('top100.json', 'r', encoding='utf-8') as f:
    hashtag_dic = json.load(f)

hashtag = {}
for i in hashtag_dic.keys():
    hashtag[i] = hashtag_dic[i]


SearchSet = []
for crawl in hashtag:
    print(crawl)
    SearchSet.append(crawl)


print(SearchSet.index('love'))
print(len(SearchSet))


#CrawlingInstagramList(SearchSet[98:], numbers=1000, max_iteration=50000)
CrawlingInstagramList(SearchSet[18:19], numbers=1000, max_iteration=10000)