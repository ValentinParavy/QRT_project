# QRT project: Can you guess the winner? 

https://challengedata.ens.fr/challenges/143 

The goal of this data challenge is to predict the outcome of football games, using real historical data at the player, team and league level. With Adriano Todisco (insert github link) we ranked 85 out of 776 on the private leaderboard, under the name of René Coty, a great man who marked French History. We present here the code that we used for our different submissions.

We divided the tasks in three parts:

- Handling the missing values
- Features Engineering
- Model selection

### Missing Values

The choice of how could we manage the missing values was crucial as nearly 32% of the samples contained at least one nan. We testes filling the nan with the average of the all other samples or only groups of similar samples, using knn, but also other methods including xgboost and bayesian bridge, where the filled values are predictions of those models. In order to compare those models, we selected a benchmark model already fitted with a dataset with the nan values dropped, we finally selected the methods that led to the best accuracy with this benchmark.


### Features Engineering

To avoid overfitting we chose to reduce the dimensionality of the data using different methods:

- PCA
- Dropping correlated features
- Features importance with a random forest

Our best scores were indeed obtained when a significant part of the features were dropped. It made even more sense to reduce the dimension as we used many logistic regressions in our predictive models that are sensitive to noisy data and high-dimensional datasets.

### Model selection

Finally once we had built of final train data we cross validated many models, including xgboost, neural networks and logistic regressions. We also mixed multiple models through a logistic regression (model ensembling). Very surprisingly, our model that scored the highest with an accuracy of $0.4875$ was actually a simple logistic regression. 

### Conclusion

At our level, our work showed the importance of features engineering and data processing and how simple models can outperform most sophisticated ones.
