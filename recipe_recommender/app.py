# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:39:24 2020

@author: choibosu
"""

import json as json
import requests
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models import DatetimeTickFormatter
from bokeh.resources import CDN
from bokeh.embed import file_html, components
from flask import Flask, render_template, request, redirect, url_for
from sklearn import base
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

class DictEncoder(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col):
        self.col = col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        def to_dict(l):
            try:
                return {x: 1 for x in l}
            except TypeError:
                return {}
        
        return X[self.col].apply(to_dict)
    
def pyfunc(x, y):
    
    x = x.todense()
    y = y.todense()
    ones = np.ones(x.shape)
    
    return (1 + np.dot(np.squeeze(np.asarray(y)), np.squeeze(np.asarray(ones)\
                                  - np.squeeze(np.asarray(x)))) ) * 1.0 / \
                                  (np.dot(np.squeeze(np.asarray(y)),\
                                          np.squeeze(np.asarray(x))) + 1)    

def recommend(Xi, Xp, Xr):
    
    sieved_recipe_df = pd.read_csv("sieved_recipes.csv") 
    sieved_recipe_df = sieved_recipe_df[sieved_recipe_df['ext_ingredients']\
                                        .map(lambda x: all(elt in x for elt in Xp))]
    sieved_recipe_df = sieved_recipe_df[sieved_recipe_df['ext_ingredients']\
                                        .map(lambda x: all(elt not in x for elt in Xr))]
    sieved_recipe_df = sieved_recipe_df.append({'ext_ingredients' : Xi}, ignore_index=True)
    ing_pipe = Pipeline([('encoder', DictEncoder('ext_ingredients')), \
                         ('vectorizer', DictVectorizer())])
    sieved_features = ing_pipe.fit_transform(sieved_recipe_df)
    sieved_nn = NearestNeighbors(n_neighbors=21, metric = lambda a, b: pyfunc(a,b)).fit(sieved_features)
    dists, indices = sieved_nn.kneighbors(sieved_features[-1])
    recommendation = sieved_recipe_df.iloc[indices[0]]
    recommendation = recommendation[['id','name','ext_ingredients']]
    
    return recommendation

def create_table(table):
    
    table = table.rename(columns = {'name': 'Dish Name', 'ext_ingredients': 'Ingredients'})
    table = table.drop([table.index[0]])
    table['Link to the Recipe'] = table.apply(lambda row: 'http://allrecipes.com/recipe/{}'\
         .format(int(row.id)), axis=1)
    table = table.drop(['id'], axis = 1) 
    
    table_html = table.to_html(classes=["table-bordered", "table-striped", "table-hover"], index = False)
    
    return table_html

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/end', methods = ['POST'])
def output():
    
    Xi = request.form['Xi']
    Xp = request.form['Xp']
    Xr = request.form['Xr']
    
    Xi = [elt.strip() for elt in Xi.split(",")]
    Xp = [elt.strip() for elt in Xp.split(",")]
    Xr = [elt.strip() for elt in Xr.split(",")]
    
    recommendation = recommend(Xi, Xp, Xr)
    result_table = create_table(recommendation)

    return render_template('end.html', script=result_table)
    
if __name__ == '__main__':
    app.run(port=33507)
    
    #example for Xi, Xp, Xr
    #onion,beef, chicken, carrot, cabbage, egg, green onion
    #shrimp
    #chicken