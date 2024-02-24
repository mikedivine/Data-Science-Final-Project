# Data Science Final Project
## Prediciting Home Values
by Kenneth Ao, Russell Frost, and Mike Divine

![Home Prices](https://assets.site-static.com/userFiles/717/image/factors-impacting-property-value.jpg)

## Introduction

The goal of this data science project is to develop a machine learning model capable of predicting home prices based on many different factors including things such as location, square footage, and the number of bedrooms and bathrooms. Predicting home prices accurately is important for buyers and sellers in the real estate market. Buyers need to know if a property is priced reasonably, while sellers aim to maximize their profit without overpricing their home.

By utilizing historical housing market data, including factors like location, property features, and previous sales prices, we will attempt to create a predictive model that provides accurate estimates of home prices. This model could assist real estate agents, buyers, and sellers in helping to make informed decisions.


## Selection of Data

[Placer County CA Residential Sales 2023](https://docs.google.com/document/d/1z7PQ3lbd_7D71Og2D5pcniy1VDgdrkDni0lkBgZGPaE/edit)

The dataset we obtained via the MLS(multiple listing service) offers comprehensive information on residential sales within Placer County, California, throughout the year 2023. It includes many variables, ranging from characteristics like neighborhood, lot size, and square footage to the number of bedrooms and bathrooms. This dataset will serve as the building block for the development of a predictive model to forecast residential property prices.


## Methods

## Tools ##

- Numpy, Pandas, Matplotlib, and Seaborn for data analysis and visualization
- Scikit-learn for inference
- Github for Version Control
- Google Colab for Collaboration and Version Control
- Inference methods used with Scikit:

## Models ##

Many models were used to test the accuracy of the machine learning algorithms.  The first model used was Linear Regression.  The predictors for this model were: bedrooms, sqft, fbathrooms, pbathrooms, and ybd.  Along with the model vizualization were used to understand the results better.  Next a Decision Tree Regressor was used on predictors bedrooms, fbathrooms, pbathrooms.  This provided us with a representation of how tree was making decisions based on these predictors.  The K Neighbors Regressor was used to determine the value for k that would yield the best results.  Forward Selection, in conjunction with Linear Regression, was also used to determine what predictors had the most effect on the machine learning model and would yield the best predictors to use for the most accurate predictions.

## Results

We did get a predictive model that got with an RMSE of $192,000 which showed we could predict prices but not near as accurate as we would like.  We feel with more data and different predictive methods we could continue to work to get closer and closer.  We had many different visualizations such as showing our predicted prices vs the Actual Prices as well as number of bedrooms vs price, number of acres vs price and a few others.  We also showed that the Listing price was very closely linked to the closing price, so this could mean the Real Estate agents have a real good handle in their market and know what a particular house should go for or it could mean them setting the lsiting value could actually affect the selling price.


## Discussion

The results showed that we have started to narrow in on the 


## Summary

Most important findings.

