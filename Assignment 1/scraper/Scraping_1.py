# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:45:39 2019

@author: Gokul
"""

from requests import get
url = 'https://www.imdb.com/search/title/?title_type=feature,tv_series&sort=num_votes,desc'
response = get(url)

from bs4 import BeautifulSoup
html_soup = BeautifulSoup(response.text, 'html.parser')
type(html_soup)

movie_containers = html_soup.find_all('div', class_ = 'lister-item mode-advanced')
#first_movie.div

names = []
years = []
imdb_ratings = []
genres = []
descriptions = []
directors = []
casts = []
# Extract data from individual movie container
for container in movie_containers:
# If the movie has Metascore, then extract:
    if container.find('div', class_ = 'ratings-metascore') is not None:
# The name
        name = container.h3.a.text
        names.append(name)
        print(name)
# The year
        year = container.h3.find('span', class_ = 'lister-item-year').text
        years.append(year)
        print(year)
# The IMDB rating
        imdb = float(container.strong.text)
        imdb_ratings.append(imdb)
        print(imdb)
        
        genre = container.find('p', class_ = 'text-muted').text
        trim_genre = genre.split('\n|\n')
        size = len(trim_genre)
        #print(trim_genre)
        if size == 3:
            genre_final = trim_genre[2]
            genres.append(genre_final.strip())
            print(genre_final.strip())
        
        
        description = container.select('p:nth-of-type(2)')
        desc = str(description[0])
        trim_desc = desc.split('<')
        trim_desc = trim_desc[1].split('>')
        descriptions.append(trim_desc[1].strip())
        print(trim_desc[1].strip())
        
        cast = container.find('p', class_ = "").text
        director = cast.split('|')
        director[0] = director[0].strip()
        director[0] = director[0].replace('\n','')
        temp2 = director[0].split(':')
        director[1] = director[1].strip()
        director[1] = director[1].replace('\n','')
        temp1 = director[1].split(':')
        print(temp2[1])
        print(temp1[1])
        casts.append(temp2[1])
        directors.append(temp1[1])
        print('\n')
        
        
        
        


