# King's County Home Price Analysis 

Contributors: Jeff Marvel, Aidan Stack, and Teigen Olson
 
## Business Problem 

 Our client is a small real estate developer looking to evaluate the viability of different projects around Seattle. To assist this effort, our team used multiple linear regression modelling to help estimate home prices, and determine which factors were the most influential on price. 
 
 ## The Data 
 
 Our data came in the form of 20,000 home sales from all over Kings County spanning from 2014-2015. It included various metrics describing each home, including overall condition, quality of the view, square footage, zipcode, etc. 
 
 ## Exploratory Data Analysis 
 
  Our first steps were to examine the dataset, find where the outliers were, and determine what to do with them. After that we looked at the correlation of each variable with our dependent variable, price. We also ran correlation matrices to check for multicollinearity issues. We sorted each variable by correlation with price and found that square footage of the living space was the most correlated. 
  
## Data Prep

 Once we had a firm grasp of what our data looked like, it was time to start manipulating it in preperation for modelling. We began by one-hot encoding various categorical variables, such as waterfront view, as well as zipcode. We found a table online that translated zipcodes for King's County into Neighborhoods, so we added those columns and one-hot encoded that feature as well. 
 
## Simple Model 
 
 For our simple model, we ran a linear regression using only square footage of living space. We chose this variable since it was the most highly correlated with price, and would serve as a good baseline for comparison with later models. 

## Modelling Iterations 

 Once we had all of our categorical variables one-hot encoded, continuous variables log transformed, and had engineered some custom features, we began the process of iterating on our original model. We tried numerous combinations of variables looking for the strongest relationship, while making sure that the variables we used would work well in serving our client's business problem. During this phase we found that while zipcode provided a slightly more accurate model than neighborhood, we decided to move forward with neighborhood as it was more applicable to our business problem. 
 
## Final Model 

 Our final model combined neighborhood, and several custom engineered features to provide us with an r-squared of .82. We were satisfied with this model because it could explain most of the variation in price, and the variabels represented actionable insights for the developer. 

## Key Takeaways 

 The old real-estate mantra 'location, location, location' holds true in our dataset. Zipcode and neighborhood served as some of our strongest predictors of overall home price. We also found that when a house takes up too much of the lot it is on, price drops. So while increases in square footage does increase price, the developer should be careful to not consume too much of the lot with living space. More bathrooms also significantly increases home price when compared to other factors. These findings combine to a global recommendation to find small homes, in premium neighborhoods, that can either be rennovated or expanded slightly. 
 
## Possible Next Steps

 While valuable, these recommendations could be improved by somehow incorporating a profitability element, as that angle is missing from our current analysis. 
 
## Presentation Link 

https://docs.google.com/presentation/d/13v98uOOZorAmWhyX4-hHCPyzTjhpOGaw1-9PSgU2f5s/edit?usp=sharing


