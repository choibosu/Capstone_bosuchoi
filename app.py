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
from bokeh.palettes import Spectral3
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

def get_plot():
    
    multiTimeline_df = pd.read_csv('multiTimeline.csv')
    multiTimeline_df = multiTimeline_df.drop([0, 1])\
                        .rename({'Category: All categories': 'Dates','Unnamed: 1': 'Stay at Home',\
                        'Unnamed: 2': 'Restaurant', 'Unnamed: 3': 'Recipe'}, axis = 1)
    multiTimeline_df.loc[multiTimeline_df['Stay at Home'] == '<1', 'Stay at Home'] = 0.5
    multiTimeline_df['Stay at Home'] = multiTimeline_df['Stay at Home'].astype(float)
    multiTimeline_df['Restaurant'] = multiTimeline_df['Restaurant'].astype(float)
    multiTimeline_df['Recipe'] = multiTimeline_df['Recipe'].astype(float)
    multiTimeline_df['Dates'] = pd.to_datetime(multiTimeline_df['Dates'])
    
    us_daily_df = pd.read_csv('us_daily.csv')
    us_daily_df['positive'] = us_daily_df['positive'].astype(float)
    us_daily_df['positiveIncrease'] = us_daily_df['positiveIncrease'].astype(float)
    us_daily_df['death'] = us_daily_df['death'].astype(float)
    us_daily_df['deathIncrease'] = us_daily_df['deathIncrease'].astype(float)
    us_daily_df['Dates'] = pd.to_datetime(us_daily_df['lastModified'])

    source = ColumnDataSource(multiTimeline_df)

    p = figure()
    p.line(source=source, x = 'Dates', y = 'Stay at Home', legend = 'stay at home')
    p.line(source=source, x = 'Dates', y = 'Restaurant', legend = 'restaurant', color=Spectral3[1])
    p.line(source=source, x = 'Dates', y = 'Recipe', legend = 'recipe', color=Spectral3[2])

    p.title.text = 'Relative Google Search Trend'
    p.xaxis.axis_label = 'Date'
    p.xaxis.formatter = DatetimeTickFormatter(days=['%m/%d'], months=['%m/%Y'], years=['%Y'])
    
    source = ColumnDataSource(us_daily_df)
    
    q = figure()
    q.vbar(source=source, x = 'Dates', top = 'positiveIncrease', width = 0.1, legend = 'Daily confirmed')
    q.vbar(source=source, x = 'Dates', top = 'deathIncrease', width = 0.1, legend = 'Daily death', color=Spectral3[1])

    q.title.text = 'Daily COVID-19 in US'
    q.xaxis.axis_label = 'Date'
    q.xaxis.formatter = DatetimeTickFormatter(days=['%m/%d'], months=['%m/%Y'], years=['%Y'])

    return (p, q)

@app.route('/')
def front():
    return render_template('front.html')

@app.route('/about')
def about():
    
    result_plot1, result_plot2 = get_plot()
    script, div = components(result_plot1)
    script2, div2 = components(result_plot2)

    return render_template('about.html', script=script, div=div, script2=script2, div2=div2) 

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