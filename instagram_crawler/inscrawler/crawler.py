from __future__ import unicode_literals

from getpass import getpass
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import glob
import json
import os
import re
import sys
import time
import traceback
from builtins import open
from time import sleep
import requests
from nltk import *
from nltk.corpus import *
from langdetect import detect
from langdetect import detect_langs

from tqdm import tqdm

from . import secret
from .browser import Browser
from .exceptions import RetryException
from .fetch import fetch_caption
from .fetch import fetch_comments
from .fetch import fetch_datetime
from .fetch import fetch_imgs
from .fetch import fetch_likers
from .fetch import fetch_likes_plays
from .utils import instagram_int
from .utils import randmized_sleep
from .utils import retry


class Logging(object):
    PREFIX = "instagram-crawler"

    def __init__(self):
        try:
            timestamp = int(time.time())
            self.cleanup(timestamp)
            self.logger = open("/tmp/%s-%s.log" % (Logging.PREFIX, timestamp), "w")
            self.log_disable = False
        except Exception:
            self.log_disable = True

    def cleanup(self, timestamp):
        days = 86400 * 7
        days_ago_log = "/tmp/%s-%s.log" % (Logging.PREFIX, timestamp - days)
        for log in glob.glob("/tmp/instagram-crawler-*.log"):
            if log < days_ago_log:
                os.remove(log)

    def log(self, msg):
        if self.log_disable:
            return

        self.logger.write(msg + "\n")
        self.logger.flush()

    def __del__(self):
        if self.log_disable:
            return
        self.logger.close()


class InsCrawler(Logging):
    URL = "https://www.instagram.com"
    RETRY_LIMIT = 10

    def __init__(self, has_screen=False):
        super(InsCrawler, self).__init__()
        self.browser = Browser(has_screen)
        self.page_height = 0

    def _dismiss_login_prompt(self):
        ele_login = self.browser.find_one(".Ls00D .Szr5J")
        if ele_login:
            ele_login.click()

    def login(self):
        browser = self.browser
        url = "%s/accounts/login/" % (InsCrawler.URL)
        browser.get(url)
        u_input = browser.find_one('input[name="username"]')
        u_input.send_keys(secret.username)
        p_input = browser.find_one('input[name="password"]')
        p_input.send_keys(secret.password)

        login_btn = browser.find_one(".L3NKy")
        login_btn.click()

        @retry()
        def check_login():
            if browser.find_one('input[name="username"]'):
                raise RetryException()

        check_login()

    def get_user_profile(self, username):
        browser = self.browser
        url = "%s/%s/" % (InsCrawler.URL, username)
        browser.get(url)
        name = browser.find_one(".rhpdm")
        desc = browser.find_one(".-vDIg span")
        photo = browser.find_one("._6q-tv")
        statistics = [ele.text for ele in browser.find(".g47SY")]
        post_num, follower_num, following_num = statistics
        return {
            "name": name.text,
            "desc": desc.text if desc else None,
            "photo_url": photo.get_attribute("src"),
            "post_num": post_num,
            "follower_num": follower_num,
            "following_num": following_num,
        }

    def get_user_profile_from_script_shared_data(self, username):
        browser = self.browser
        url = "%s/%s/" % (InsCrawler.URL, username)
        browser.get(url)
        source = browser.driver.page_source
        p = re.compile(r"window._sharedData = (?P<json>.*?);</script>", re.DOTALL)
        json_data = re.search(p, source).group("json")
        data = json.loads(json_data)

        user_data = data["entry_data"]["ProfilePage"][0]["graphql"]["user"]

        return {
            "name": user_data["full_name"],
            "desc": user_data["biography"],
            "photo_url": user_data["profile_pic_url_hd"],
            "post_num": user_data["edge_owner_to_timeline_media"]["count"],
            "follower_num": user_data["edge_followed_by"]["count"],
            "following_num": user_data["edge_follow"]["count"],
            "website": user_data["external_url"],
        }

    def get_user_posts(self, username, number=None, detail=False):
        user_profile = self.get_user_profile(username)
        if not number:
            number = instagram_int(user_profile["post_num"])

        self._dismiss_login_prompt()

        if detail:
            return self._get_posts_full(number)
        else:
            return self._get_posts(number)

    def get_latest_posts_by_tag(self, tag, num):
        url = "%s/explore/tags/%s/" % (InsCrawler.URL, tag)
        self.browser.get(url)
        return self._get_posts_location(num)

    def auto_like(self, tag="", maximum=1000):
        self.login()
        browser = self.browser
        if tag:
            url = "%s/explore/tags/%s/" % (InsCrawler.URL, tag)
        else:
            url = "%s/explore/" % (InsCrawler.URL)
        self.browser.get(url)

        ele_post = browser.find_one(".v1Nh3 a")
        ele_post.click()

        for _ in range(maximum):
            heart = browser.find_one(".dCJp8 .glyphsSpriteHeart__outline__24__grey_9")
            if heart:
                heart.click()
                randmized_sleep(2)

            left_arrow = browser.find_one(".HBoOv")
            if left_arrow:
                left_arrow.click()
                randmized_sleep(2)
            else:
                break

    def _get_posts_full(self, num):
        @retry()
        def check_next_post(cur_key):
            ele_a_datetime = browser.find_one(".eo2As .c-Yi7")

            # It takes time to load the post for some users with slow network
            if ele_a_datetime is None:
                raise RetryException()

            next_key = ele_a_datetime.get_attribute("href")
            if cur_key == next_key:
                raise RetryException()

        browser = self.browser
        browser.implicitly_wait(1)
        ele_post = browser.find_one(".v1Nh3 a")
        ele_post.click()
        dict_posts = {}

        pbar = tqdm(total=num)
        pbar.set_description("fetching")
        cur_key = None

        # Fetching all posts
        for _ in range(num):
            dict_post = {}

            # Fetching post detail
            try:
                if cur_key is not None:
                    check_next_post(cur_key)

                # Fetching datetime and url as key
                ele_a_datetime = browser.find_one(".eo2As .c-Yi7")
                cur_key = ele_a_datetime.get_attribute("href")
                dict_post["key"] = cur_key
                fetch_datetime(browser, dict_post)
                fetch_imgs(browser, dict_post)
                fetch_likes_plays(browser, dict_post)
                fetch_likers(browser, dict_post)
                fetch_caption(browser, dict_post)
                fetch_comments(browser, dict_post)

            except RetryException:
                sys.stderr.write(
                    "\x1b[1;31m"
                    + "Failed to fetch the post: "
                    + cur_key
                    + "\x1b[0m"
                    + "\n"
                )
                break

            except Exception:
                sys.stderr.write(
                    "\x1b[1;31m"
                    + "Failed to fetch the post: "
                    + cur_key
                    + "\x1b[0m"
                    + "\n"
                )
                traceback.print_exc()

            self.log(json.dumps(dict_post, ensure_ascii=False))
            dict_posts[browser.current_url] = dict_post

            pbar.update(1)
            left_arrow = browser.find_one(".HBoOv")
            if left_arrow:
                left_arrow.click()

        pbar.close()
        posts = list(dict_posts.values())
        if posts:
            posts.sort(key=lambda post: post["datetime"], reverse=True)
        return posts



    def _get_posts_location(self, num):
        # @retry()
        def language_detect(sentence):
            def lang_ratio(input):
                lang_ratio = {}
                tokens = wordpunct_tokenize(input)
                words = [word.lower() for word in tokens]
                for language in stopwords.fileids():
                    stopwords_set = set(stopwords.words(language))
                    words_set = set(words)
                    common_elements = words_set.intersection(stopwords_set)
                    lang_ratio[language] = len(common_elements)
                return lang_ratio

            def detect_language(input):
                ratios = lang_ratio(input)
                language = max(ratios, key=ratios.get)
                # print(language)
                return language

            # print(sentence.replace("#", ""))
            return detect_language(sentence.replace("#", ""))

        def check_next_post(cur_key):
            ele_a_datetime = browser.find_one(".eo2As .c-Yi7")

            # It takes time to load the post for some users with slow network
            if ele_a_datetime is None:
                print("datetime error ", cur_key)
                browser.implicitly_wait(5)
                # ele_post = browser.find_one(".v1Nh3 a")
                # ele_post.click()

                while ele_a_datetime is None:
                    left_arrow = browser.find_one(".HBoOv")
                    # print(left_arrow)
                    if left_arrow:
                        left_arrow.click()
                    ele_a_datetime = browser.find_one(".eo2As .c-Yi7")
                    # print(ele_a_datetime)
                    if ele_a_datetime is not None:
                        break
                # raise Exception()

            next_key = ele_a_datetime.get_attribute("href")
            if cur_key == next_key:
                print("cur=next error:", cur_key)
                while cur_key == next_key:
                    left_arrow = browser.find_one(".HBoOv")
                    # print(left_arrow)
                    if left_arrow:
                        left_arrow.click()
                    ele_a_datetime = browser.find_one(".eo2As .c-Yi7")
                    # print(ele_a_datetime)
                    if ele_a_datetime is not None:
                        next_key = ele_a_datetime.get_attribute("href")

                raise Exception()

        browser = self.browser
        browser.implicitly_wait(1)
        ele_post = browser.find_one(".v1Nh3 a")
        ele_post.click()
        dict_posts = {}

        pbar = tqdm(total=num)
        pbar.set_description("fetching")
        cur_key = None

        # Fetching all posts
        # for i in range(num):
        i = 0
        while True:
            if i == 100:
                break
            dict_post = {}
            # Fetching post detail
            try:
                check_next_post(cur_key)
                # print(cur_key)
                # Fetching datetime and url as key
                ele_a_datetime = browser.find_one(".eo2As .c-Yi7")
                cur_key = ele_a_datetime.get_attribute("href")
                fetch_caption(browser, dict_post)
                if dict_post["hashtags"] is None:
                    # print("no hashtags")get_latest_posts_by_tag
                    raise Exception()
                language = detect(dict_post['caption'].replace("#", ""))
                # print(dict_post['caption'])
                # if language == 'en':
                #     language = 'english'
                # language = language_detect(dict_post['caption'])
                if language != 'en':
                    # print(dict_post['caption'])
                    # pbar.update(1)
                    # left_arrow = browser.find_one(".HBoOv")
                    # if left_arrow:
                    #     left_arrow.click()
                    # num = num + 1
                    # continue
                    # print("language: ", language)
                    # print(dict_post['caption'])

                    raise Exception()
                # else:
                    # print(dict_post['caption'])
                    # print(detect_langs(dict_post['caption']))

                # try:
                    # Location url and name
                    # location_div = browser.find_element_by_class_name('M30cS').find_elements_by_tag_name('a')
                location_div = browser.find_one(".M30cS a")
                """
                # print(location_div)
                if location_div is None:
                    # left_arrow = browser.find_one(".HBoOv")
                    # if left_arrow:
                    #     left_arrow.click()
                    # continue
                    # print("no location")
                    raise Exception()
                print('!!!!!!!!!!!!!!!!!!!!!!!')
                print(location_div)
                if location_div is not None:
                    location_url = location_div.get_attribute('href')
                    location_name = location_div.text
                    #location_url = location_div[0].get_attribute('href')
                    #location_name = location_div[0].text
                    # Longitude and latitude
                    location_id = location_url.strip('https://www.instagram.com/explore/locations/').split('/')[0]
                    url = 'https://www.instagram.com/explore/locations/' + location_id + '/?__a=1'
                    url2 = "https://www.instagram.com/accounts/login/?next=/explore/locations/{0}/?__a=1".format(location_id)
                    #url = "https://www.instagram.com/accounts/login/?next=/explore/locations/{0}".format(location_id)



                    login_url = 'https://www.instagram.com/'
                    with requests.session() as c:
                        c.get(login_url)
                        token = c.cookies['csrftoken']
                        print('++++++++++++++')
                        print(token)

                    LOGIN_INFO = {
                        'username': '01090173763',
                        'password': 'tjdwls12',
                        'access_token': token
                    }

                    # Session 생성, with 구문 안에서 유지
                    with requests.Session() as s:
                        # HTTP POST request: 로그인을 위해 POST url와 함께 전송될 data를 넣어주자.
                        login_req = s.post('https://www.instagram.com/accounts/login', data=LOGIN_INFO, allow_redirects=True)
                        # 어떤 결과가 나올까요?
                        print('=================')
                        print(login_req.status_code)

                        response = s.get(url)
                        print(response.url)
                        data=json.loads(json.dumps(response.text))
                        print(response.text)
                        lat = data['graphql']['location']['lat']
                        lng = data['graphql']['location']['lng']


                        location_dic = dict()
                        location_dic['name'] = location_name
                        location_dic['url'] = location_url
                        location_dic['id'] = location_id
                        location_dic['lat'] = lat
                        location_dic['lng'] = lng
                        dict_post['location'] = location_dic

                    # print(location_dic)
                
                else:
                    num = num + 1
                    continue
                """
                # except Exception:
                #
                #     continue
                dict_post["key"] = cur_key
                fetch_datetime(browser, dict_post)
                fetch_imgs(browser, dict_post)
                # fetch_likes_plays(browser, dict_post)
                # fetch_likers(browser, dict_post)

                # fetch_comments(browser, dict_post)

                # post = browser.find_element_by_class_name('ltEKP')


            # except RetryException:
            #     sys.stderr.write(
            #         "\x1b[1;31m"
            #         + "Failed to fetch the post: "
            #         + cur_key
            #         + "\x1b[0m"
            #         + "\n"
            #     )
            #     num = num + 1
            #     print("copm")
            #     continue

            except Exception:
                # sys.stderr.write(
                #     "\x1b[1;31m"
                #     + "Failed to fetch the post: "
                #     + str(cur_key)
                #     + "\x1b[0m"
                #     + "\n"
                # )
                # traceback.print_exc()
                num = num + 1
                # print("DF?")
                # pbar.update(1)
                browser.implicitly_wait(1)
                left_arrow = browser.find_one(".HBoOv")
                # print(left_arrow)
                if left_arrow:
                    left_arrow.click()
                continue

            self.log(json.dumps(dict_post, ensure_ascii=False))
            dict_posts[browser.current_url] = dict_post

            pbar.update(1)
            left_arrow = browser.find_one(".HBoOv")
            if left_arrow:
                left_arrow.click()
            i = i + 1
            # print(cur_key)
        print("Looked up ", num, "posts total")
        pbar.close()
        posts = list(dict_posts.values())
        if posts:
            posts.sort(key=lambda post: post["datetime"], reverse=True)
        return posts

    def _get_posts(self, num):
        """
            To get posts, we have to click on the load more
            button and make the browser call post api.
        """
        TIMEOUT = 600
        browser = self.browser
        key_set = set()
        posts = []
        pre_post_num = 0
        wait_time = 1

        pbar = tqdm(total=num)

        def start_fetching(pre_post_num, wait_time):
            ele_posts = browser.find(".v1Nh3 a")
            for ele in ele_posts:
                key = ele.get_attribute("href")
                if key not in key_set:
                    dict_post = {}
                    ele_img = browser.find_one(".KL4Bh img", ele)
                    caption = ele_img.get_attribute("alt")
                    img_url = ele_img.get_attribute("src")
                    fetch_caption(browser, dict_post)
                    key_set.add(key)
                    posts.append({"key": key, "caption": caption, "hashtags": dict_post, "img_url": img_url})
            if pre_post_num == len(posts):
                pbar.set_description("Wait for %s sec" % (wait_time))
                sleep(wait_time)
                pbar.set_description("fetching")

                wait_time *= 2
                browser.scroll_up(300)
            else:
                wait_time = 1

            pre_post_num = len(posts)
            browser.scroll_down()

            return pre_post_num, wait_time

        pbar.set_description("fetching")
        while len(posts) < num and wait_time < TIMEOUT:
            post_num, wait_time = start_fetching(pre_post_num, wait_time)
            pbar.update(post_num - pre_post_num)
            pre_post_num = post_num

            loading = browser.find_one(".W1Bne")
            if not loading and wait_time > TIMEOUT / 2:
                break

        pbar.close()
        print("Done. Fetched %s posts." % (min(len(posts), num)))
        return posts[:num]
