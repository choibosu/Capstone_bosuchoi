# Capstone_bosuchoi

This project is to build a recipe recommendation system web application using Flask.
Recipes are scraped and parsed by "parse-recipes_modified.py" from allrecipes.com, and are manipulated and saved in "sieved_recipes.csv"
This csv file contains only partial recipes that are used in the app due to the file size limit (25mb) in GitHub.
The parsing code is originally from recipe-parser" of "kbrohkahn" but modified based on austhor's preference.
The web application is deployed to "Heroku" and found in "https://capstone-bosuchoi.herokuapp.com/".
