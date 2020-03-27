import wikipediaapi

wiki = wikipediaapi.Wikipedia( language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
p_wiki = wiki.page("apple")
print(p_wiki.text)

"""""
with open("apple.txt", "w") as f:
    f.write(p_wiki.text)
"""""