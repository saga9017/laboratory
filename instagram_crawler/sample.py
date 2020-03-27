import json


a={'a':1, 'b':2}

print(json.loads('s'))

### Instagram Turbo ###
import requests

wanted_username = 'justinbieber'

check = requests.get('https://www.instagram.com/' + wanted_username + '/')

print(check.url)
answer = check.ok
print(answer)


def login():
    login_url = 'https://www.instagram.com/'
    with requests.session() as c:
        c.get(login_url)
        token = c.cookies['csrftoken']
        print(token)

        payload = {
            'username': '*****',
            'password': '*****',
            'access_token': token
        }

        post = c.post(login_url, data=payload, allow_redirects=True)

        edit = c.get('https://www.instagram.com/accounts/edit/')
        print(edit.url)


login()