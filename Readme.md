A Data-driven Approach to Predict Scores in T20

Cricket Match Using Machine Learning Classifier

Abstract

Accurate score prediction is essential for teams to develop winning strategies because of the growing popularity of T20 cricket and the significance of setting a challenging target in the first innings. The suggested method entails gathering his- torical information on T20 matches and applying feature engineering approaches to extract pertinent features. To forecast the first innings score, various regres- sion methods, like XGBoost regression, Lasso regression, and Ridge regression are trained on the dataset. Metrics such as mean absolute error, root mean squared error, and R-squared values are used to assess the performance of the models. The findings demonstrate the potential of machine learning techniques for pre- dicting the first innings score in T20 cricket matches, offering useful information for team strategy. The developed models, implemented codes, and user interface designs are deployed in this link: h[ttps://github.com/AST-TheCoder/T20](https://github.com/AST-TheCoder/T20)

Keywords: Score, Prediction, T20, RGBoost, Lasso, Ridge, Addable Score, CRR, Innings

1 Introduction

Cricket is being popular day by day. It is high time to be more accurate about predicting the scores of each inning. Predicting scores can be helpful to furnish match strategies. Also, it will increase the attraction of matches to the audience. In T20 cricket matches, predicting the first innings score is a valuable and difficult topic that has important practical applications for teams, coaches, and broadcasters. It can be increased The efficacy and accuracy of these predictions using machine learning techniques like XGBoost, Lasso, and Ridge regression, allowing for better decision- making and boosting the fan experience while watching. It is obvious that cricket is a highly unpredictable game. The model has been built as efficiently as possible.

Using machine learning methods to forecast the results of T20 cricket matches has gained popularity in recent years. Large data sets can be analyzed using machine learn- ing to find patterns that can be used to forecast future events. The challenge is to use machine learning techniques to forecast the first innings score in T20 cricket matches. The goal is to use historical data to simulate the relationship between the batting team’s performance and the final score and to assess how well XGBoost, Lasso, and Ridge regression techniques perform in this setting. Insufficient comparative examina- tion of machine learning techniques, such as XGBoost, Lasso, and Ridge regression, for forecasting the first innings score in T20 cricket matches is the research needs that are identified. The performance of multiple algorithms needs to be thoroughly eval- uated and compared in order to determine the most efficient strategy, even if prior studies have used a variety of models to tackle this issue.

The scores depend on a lot of variables like teams, squads, venues, weather, etc [\[1](#_page11_x129.58_y565.51)]. As The model is going to predict the live score, it depends on the data of the last five overs, number of wickets fallen, number of overs done, current score, venue, and the teams. This paper is going to propose three types of Regression models to predict which are XGBoost regression, Lasso regression, and Ridge regression. And, XGBoost regression performed better than others with almost 99% accuracy. Though The model has some limitations. Such as At least five overs have to be played first, it

will not declare the weather to predict, and match termination points are not learned properly.

The rest of this paper is structured as follows. A summary of related research is given in the area of T20 cricket score prediction in Section 2. The dataset and the features employed for prediction are discussed in Section 3. Also, the machine learning models that are utilized for prediction are discussed in the same Section. The findings of the experiments and the contrast proposed strategy with other strategies are discussed in Section 4. The paper is wrapped up in Section 5.

2 Related Works

The topic of predicting cricket scores is not that much common yet. A few popular pieces of research are described below.

The authors have created a web application that does predictive analysis of a live T-20 match in order to forecast the result and the match’s outcome before it starts. The app also has a live score scraper that collects precise match data and feeds it into the prediction algorithm. The application lets you view the forecasts as graphs for a more thorough understanding. These predictions are made using algorithms like Multivariate Polynomial Regression and Random Forest Classifier, and the web application is created with Flask [1][.](#_page11_x129.58_y565.51)

A promising field for research is cricket. Cricket is a sport that lends itself well to statistical analysis and machine learning because of its rising popularity and financial rewards. The dynamic nature of cricket and the intricate laws that govern it make the endeavor difficult. Due to the disparities in the techniques, neither the numerous approaches adopted nor the information given from the existing work are very clear nor effectively recorded. Future studies will be aided by carefully analyzing and doc- umenting the benefits and downsides of the existing work. This study is the outcome of ongoing research; by the time the research is complete [2].

Cricket is no exception to the rule that player selection is one of the most crucial responsibilities in any sport. The rival team, the setting, the player’s present form, etc. all have an impact on how well a player performs. From a roster of 15 to 20 players, the captain, coach, and team management choose 11 players for each game. To choose the best starting 11 for each game, they examine various player traits and statistics. Each batter contributes by going for the highest possible run total, while each bowler contributes by getting the most wickets and giving up the fewest runs. This essay is an attempt to forecast player performance by estimating how many runs each batsman will achieve and how many wickets each bowler will claim for both sides. Both issues are intended to be classed as problems, where the ranges for the number of runs and wickets are different. To create the prediction models for both issues, the paper employed decision tree classifiers, random forest, multiclass SVM, and Naive Bayes classifiers. The most accurate classifier for both issues was discovered to be Random Forest [[3\].](#_page11_x129.58_y657.43)

Predicting the future sounds like magic, whether it is determining where the price of a stock is going to end up or anticipating a potential customer’s intent to buy a company’s items. It has a significant advantage if something’s future can be forecasted with any degree of accuracy. The enchantment has been amplified and the enigma has been revealed thanks to machine learning. It has also been useful in the sporting domains. The sport of cricket is adored by billions of people who eagerly await the results. In this essay, the seventh T20 World Cup, which took place in Australia in 2020 was explored. For the purpose of predicting the winner of the T20 cricket world cup, The paper compared widely used machine learning algorithms. Using a unique accu- racy metric, Random Forest emerged as the top machine learning algorithm among the produced models. A custom accuracy of 80.86% was attained. According to this forecast, Australia will win the 2020 T20 world championship. The ESPN Cricinfo dataset has been used for this purpose [4].

Currently, the first innings score in One Day International (ODI) cricket matches is forecasted using the Current Run Rate, which is the number of runs scored per a total number of overs bowled. It excludes elements like the number of wickets lost and

the match’s location. Furthermore, there is no way to anticipate the result of the game in the second innings. In this research, a model with two methodologies is provided. The first forecasts the score of the first innings not only based on the current run rate but also taking into account the number of wickets lost, the location of the game, and the batting team. The second approach forecasts the result of the game in the second inning taking into account the same factors as the first approach as well as the aim set for the batting team. For the first and second innings, respectively, these two techniques have been put into practice utilizing Linear Regression Classifier and Naive Bayes Classifier. The 50 overs of the game were divided into 5 over intervals for both techniques and for each interval, the aforementioned characteristics were recorded for all non-curtailed matches played between 2002 and 2014 by each team independently. It was discovered from the findings that the accuracy of Nave Bayes in forecasting match outcomes increased from 68% initially from 0–5 overs to 91% till the end of the 45th over. Error in Linear Regression classifier is less than Current Run Rate method in estimating the final score [5[\].](#_page11_x129.58_y749.36)

Sports analytics is one application where the use of machine learning combined with data mining techniques has become a popular area of study. With a net fan base of almost 2.5 billion, cricket is one of the most popular sports in Australia, the Caribbean, the United Kingdom, and South Asian countries. More than 100 different countries send large numbers of spectators to watch the game, and the general public is quite interested in forecasting the results. Numerous pre-game and in-game fac- tors influence how a cricket match turns out. A match’s outcome is mostly influenced by pre-game factors like the location, historical performance data, innings (first/sec- ond), team strength, etc., and different in-game factors like the toss, run rate, wickets remaining, strike rate, etc. In this study, Decision Trees and Multilayer Perceptron Networks, two distinct machine learning (ML) methodologies, have been utilized to analyze the impact these various elements had on the result of a cricket match. CricAI: Cricket Match Outcome Prediction System was created in light of these findings. The pregame factors considered by the created method for predicting the outcome of a spe- cific match include the field, venue (home, away, neutral), and innings (first/second) [\[6](#_page12_x105.05_y105.63)].

A short version of cricket known as T20 is commonly abbreviated to T20. Each of the two teams in a Twenty20 match includes 11 players, and each inning lasts for 20 overs. One of the factors contributing to this cricket’s rapid appeal is the fact that it is a very variable species. A model with two approaches is presented in this study, the first of which predicts the first innings score based on the current run rate, the number of wickets lost, the match location, and the batting team. The second technique predicts the result of the game in the second innings using the same factors as the first method in addition to the batting team’s objective. XGBoost Regression and for the first and second innings, respectively [7].

3 Methodology

In this study, machine learning approaches are used to forecast the first innings score in T20 cricket matches. The three well-liked regression techniques XGBoost, Lasso, and Ridge regression are implemented specifically to estimate the correlation between the historical performance of the batting team and the ultimate result. The models are trained and tested using a dataset of T20 cricket matches that includes information about the number of runs, wickets, overs, and venue. Missing values, scaling the features, and encoding categorical variables are handled as part of the preprocessing of the data. The hyperparameters of each algorithm are then adjusted using a grid search, and the top-performing algorithm is chosen based on its stability and predicted accuracy. Utilizing metrics like R-squared, mean absolute error, and root mean squared error, the models are assessed. With the use of the technique, It can be assessed how well each algorithm performs in terms of predicting T20 cricket scores.

1. Data analysis

The project began with a dataset containing information from more than 1400 T20 matches. Gender-specific filters are applied to the dataset. Because it is a model to forecast the first innings score of a men’s T20 match. The number of teams who played fewer than five matches has also been noted. The accuracy will be impacted by these matches. The information of other teams was not targeted; only the well-known teams were. Due to the DLS system or inclement weather, some of the matches are not completed also. The matches with these issues are filtered.

The full name of each stadium is included in the dataset. A lengthy string, however, can harm regression models. Since all stadium names begin with the city’s name, it has been attempted to identify the city from the venue name. The dataset contains eight variables, including the batting team, bowling team, city, balls left, wickets left, current run rate, and runs scored in the previous five overs. Every 120 to 130 rows, beginning with the first row, contain the ball-by-ball information for each match. As it is supervised learning, it is applied to every given match.

Additionally, dropping those balls precisely prior to the extra balls has been attempted, however, the run has been modified. It denotes that all of the additional balls’ data has been merged with that of the last valid ball. Also included are 30 additional rows using the same data as the match-ending row. Understanding the ter- mination circumstances of a match for a model will improve accuracy. The model will begin forecasting after the game is done until there are at least five overs, depending on the variables. There is now a promising score. It is not necessary to forecast the clear results scored. It will forecast the potential score that could be added through- out the remaining overs. The independent variable for the dataset is addible scores. The dataset ultimately contains information on more than 400+ matches. Theoreti- cally, 150 rows are required to express each match. The dataset now has more than 60000 rows. A sample of the completed dataset is shown in Fig. 1.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.001.png)

Fig. 1: Sample of the final dataset

In comparison to regression models like Lasso regression or Ridge regression, this paper enables us to employ XGBooost regression to obtain better results. First off, the models don’t really anticipate the final result. since it is evident that the runs scored thus far. It should not be recalculated. The first-inning final scenarios have been predicted in many ways. Such as,

1. Final Run Rate: The final run rate can be predicted. The final score can be calculated using Equation 1 [5[\].](#_page11_x129.58_y749.36)

Predictedscore = 20 ∗PredictedRunRate (1)

Predicting the ultimate run rate is very similar to predicting the final score. Addi- tionally, it has certain accuracy issues. There is a problem that significantly affects accuracy. Let’s examine the graph in Figure 2.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.002.png)

Fig. 2: Ratio of predicted runs vs actual runs

You can observe that if there are fewer runs or more runs, the amount of data is decreasing. If some matches’ current data falls inside this range, the forecasted result will be tainted. There are many ups and downs in run rate as well. Prediction is not effective.

2. Addible Score within remaining overs: This is also another way to predict the final state. Predict an addible score, and using Equation 2 the final predicted score can be calculated.

Predictedscore = CurrentScore + PredictedAddibleScore (2)

Also, You will get an equal density of data for any current state of the match. Look at the graph(Fig. 3).

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.003.png)

Fig. 3: Ratio of predicted addible runs vs actual addible runs

Fig. 3 is showing the consistency of data. It is time to predict the addible runs using the machine learning models described below.

2. XGBoost Regression

Popular machine learning algorithms for regression tasks include XGBoost (Extreme Gradient Boosting). In order to increase the precision and generalization of the final model, it is an extension of the gradient boosting method that combines the predictions of various weak regression models.

In XGBoost regression, a number of decision trees are successively trained, with each tree picking up on the mistakes of the one before it. The approach updates the model parameters while minimizing a loss function via gradient descent optimization. The predictions from each tree are combined into the final model using weights based on how important they are in lowering the overall inaccuracy.

Due to XGBoost’s strong performance and scalability, which allow it to deal with big datasets and high-dimensional feature spaces, it has gained popularity. To enhance the performance of the model, it also provides a variety of objective functions, regular- ization strategies, and hyperparameter tuning choices. All things considered, XGBoost regression is a potent machine learning method that can be applied to a variety of regression tasks, including determining the score of the first innings in T20 cricket matches[[8\].](#_page12_x105.05_y197.55)

In Fig. 4, The pipeline is shown for the XGBoost regression. At first, The columns with non-numerical data are selected and passed through the Onehotencoder. Then, It has been scaled. Lastly, XGBRegressor starts predicting.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.004.png)

Fig. 4: XGBoost Regression Pipeline

3. Lasso Regression

A regularization method for linear regression models is called Lasso (Least Absolute Shrinkage and Selection Operator) regression. By punishing the absolute value of the model coefficients and decreasing them towards zero, it seeks to lessen overfitting. In Lasso regression, the L1 penalty can make some coefficients exactly zero, thus deleting the corresponding features from the model. This can promote sparsity. Because Lasso can determine the most crucial characteristics of the target variable, it is helpful for feature selection and interpretation. The regularization parameter lambda, which must be adjusted via cross-validation, determines the severity of the penalty. Machine learning frequently employs lasso regression, particularly when working with high- dimensional datasets and sparse models[9].

In Fig. 5, The pipeline is shown for the Lasso regression. At first, The columns with non-numerical data are selected and passed through the Onehotencoder. Then, It has been scaled. Then, the data runs through the GridSearchCV for the best fit of alpha parameters. Lastly, the Lasso regressor starts predicting.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.005.png)

Fig. 5: Lasso Regression Pipeline

4. Ridge Regression

When regularizing linear regression models, ridge regression adds a penalty com- ponent to the least squares objective function in an effort to decrease overfitting. The penalty term is determined by multiplying a regularization parameter called alpha by the squared magnitude of the model coefficients. Without enforcing sparsity like Lasso, the L2 penalty in Ridge regression smooths the model coefficients by decreasing them towards zero. When the predictor variables are highly correlated, multicollinearity, a typical issue in linear regression, is lessened as a result. The regularization parameter alpha regulates the severity of the penalty, and cross-validation is required to fine- tune it. In machine learning, ridge regression is frequently employed, especially when working with small to medium-sized datasets. Lowering the variance can enhance the generalization capabilities of the model[10[\].](#_page12_x100.07_y277.48)

In Fig. 6, the pipeline for ridge regression is working the same as lasso regression.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.006.png)

Fig. 6: Ridge Regression Pipeline

5. Data Flow

The data has been processed as indicated in the data analysis section, as can be seen from the above graphic (Fig. 7). The required features for prediction are extracted after the dataset has been filtered. The dataset is then divided into train and test data. First, an attempt was made to divide them based on Match ID. Some complete

matches were used to train the model. then made an effort to forecast others. But the accuracy was lacking. The dataset was then divided at random between each ball. After that, pipelines are developed to forecast how a game’s opening innings will turn out. It is producing better results.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.007.png)

Fig. 7: Data Flow Diagram

6. User Interface

This part is a short description of the architecture the user interface should have. The user will have three string options. Those are the batting team, bowling team, and venue. Then They give the input of the current score, overs played, wickets that have fallen, and the last five overs’ runs. Then the necessary terms are calculated using Equations 3, 4, and 5 written below.

BallsRemaining = 120 − 6 ∗OversPlayed (3) WicketsRemaining = 10 − Wicketsthathavefallen (4)

CurrentScore

CurrentRunRate = (5)

OversPlayed

After calculating all of the data, the prediction has been started through the gen- erated pipeline by regression models. The sample of the user interface can be built as shown in Fig. 8.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.008.png)

Fig. 8: Sample of the user interface to take input and show output

4 Results and Discussion

Three accuracy measuring matrices are implemented and those are r-square score, mean absolute error, and root mean squared error. And also a distribution plot is shown to understand the variance between real data and predicted data.

Here r-squared score measures the fitness of the model according to reality. If it is 0, the model does not fit. If it is 1, it means that the model fits perfectly. Equation 6 is representing the r-square score[[11\].](#_page12_x100.07_y311.44)

r2 = 1 − (yi − yˆ)2 (6)

(yi − y¯)2

where,

y¯ = meanvalueof actualdata.

yˆ = predictedvalue.

The mean absolute error measures the mean distance of data from the regression line using Equation 7 below[[11\].](#_page12_x100.07_y311.44)

1 N

MAE = N i − yˆ|

|y (7) i=1

The root mean squared error is the standard deviation of actual data and predicted data, which is calculated by Equation 8 [11[\].](#_page12_x100.07_y311.44)![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.009.png)

1 N

RMSE = N (yi

− yˆ)2 (8)

i=1

Also, The change in the data distribution is shown by a distribution plot or Dist- plot. The total distribution of continuous data variables is represented by a Seaborn Distplot[[12\].](#_page12_x100.07_y357.40) The blue line is showing predicted data whereas the red line is actual data.

1. Output of Lasso regression

The score obtained by lasso regression is satisfying. The R-square score is enough. But, not that much promising. The model worked its best when the value of the alpha parameter is 40.

r2score = 0.884106 MAE = 13.527113 RMSE = 19.264971

and the distribution plot is shown in Fig. 9 of Lasso regression. It is showing that the distortion is very high with the increment of runs.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.010.png)

Fig. 9: Distribution Plot of Lasso Regression output

2. Output of Ridge regression

The score obtained by ridge regression is also satisfying. The R-square score is less than the lasso regression. But, the difference is very tiny. Also, the model worked its best when the value of the alpha parameter is 40. It can be said that ridge regression and lasso regression behaved almost the same.

r2score = 0.884095 MAE = 13.528754 RMSE = 19.265946

and the distribution plot is shown in Fig. 10 of ridge regression. Also, It can be noticed in Fig. 9, The behavior of distortion is the same as lasso regression.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.010.png)

Fig. 10: Distribution Plot of Ridge Regression output

3. Output of XGBoost regression

XGBoost regression is performing outstandingly. It is the highest score till now. The parameters to get an efficient prediction is n~~ estimator = 1000, learning rate = 0.2, and max depth = 12.

r2score = 0.990038 MAE = 2.196730 RMSE = 5.648247

and the distribution plot is shown in Fig. 11 of XGBoost regression. Fig. 11 shows that the distortion is too low to notice. But, yet cricket is a very unpredictable game.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.011.png)

Fig. 11: Distribution Plot of XGBoost Regression output

The actual data and the data predicted by XGBoost regression of some matches are also plotted in Fig. 12 to compare side by side. 25 matches are taken to predict randomly. And Fig. 12 shows that the actual and predicted runs are almost the same.

![](img\Aspose.Words.c36fc0ec-b6f7-4a50-aa52-5bdc6c64d5b7.012.png)

Fig. 12: Actual and predicted data of some matches

In the paper, ”Live Cricket Score Prediction Web Application using Machine Learning”, the model was implemented by multivariate polynomial regression and ran- dom forest with accuracy 67.3% and 55% respectively [1].[ In](#_page11_x129.58_y565.51) the paper, ”CRICKET SCORE PREDICTION USING XGBOOST REGRESSION”, the implemented model is XGBoost regression also. And the pipeline can achieve 97.39% accuracy at most [\[7](#_page12_x105.05_y163.59)]. Whereas, the pipeline achieved almost 99% accuracy in implementing XGBoost regression using more effective parameters in this paper. In the paper, ”Score and Win- ning Prediction in Cricket through Data Mining”, the accuracy is promising though it is about ODI cricket. The accuracy is 70% till 0 - 5 overs and The accuracy goes

up to 91% for the next 45 overs [5[\]. ](#_page11_x129.58_y749.36)Also in this paper, the pipelines are giving 88.4%, 88.4%, and 99% for lasso, ridge, and XGBoost regressions respectively.

Statistics demonstrate that XGBoost regression has been more effective thus far. The model still has several flaws, though. Cricket frequently maintains certain unusual states because it is unpredictable. similar to how 200+ runs can be scored in the best leagues. However, in international matches, the players find it difficult to score 200 runs or more. In these type match situations, the dataset has a lower density. The model begins to act a little woozy if the input is these kinds of states.

An issue is to make the model understand the match termination logic. It can add some runs to the current score when you ask to predict after the fifteenth over with ten wickets fallen or after twenty overs done. But it works perfectly till the match is not finished. These issues are being tried to solve.

The prediction can be more accurate using some more variables which can help to predict outcomes such as weather, pitch condition, etc. Also, the work can be advanced to the second innings. The winning probability of the batting team depending on the target can be predicted. The remaining balls and wickets can be also predicted if the second-inning batting team is going to win. If the other team is going to win, then the runs left can be predicted. It will help to predict the net run rate, an accurate series winner, etc.

There is a lot of work yet to be done. In the upcoming research, it will be tried to look into how to predict T20 cricket scores using other features like player statistics and performance history. To further increase the precision of the predictions, it is also intended to investigate the usage of ensemble models and other cutting-edge machine-learning strategies.

5 Conclusion

In this research, utilizing the XGBoost Regression model, a machine learning strat- egy is developed to forecast the score of a T20 cricket match’s first innings. The model is trained to predict the score based on many characteristics such as team makeup, venue, current run rate, etc. using a dataset of historical T20 cricket match scores. The findings showed that the machine learning method can accurately forecast the result of a T20 cricket match. Compared to Lasso and Ridge regression, the XGBoost regression model outperformed other models in terms of accuracy and processing effi- ciency. The findings show the promise of machine learning methods for forecasting T20 cricket scores, and they may be helpful to coaches, bookies, and cricket fans.

Overall, the work opens up new directions for research in this area and shows how machine learning may be used to forecast T20 cricket scores.

References

1. Mundhe,<a name="_page11_x129.58_y565.51"></a> E., Jain, I., Shah, S.: Live cricket score prediction web application using machine learning. In: 2021 International Conference on Smart Generation Computing, Communication and Networking (SMART GENCON) (2021). IEEE
1. Hatharasinghe,<a name="_page11_x129.58_y611.47"></a> M.M., Poravi, G.: Data mining and machine learning in cricket match outcome prediction: missing links. In: 2019 IEEE 5th International Conference for Convergence in Technology (I2CT), pp. 1–4 (2019). IEEE
1. Passi,<a name="_page11_x129.58_y657.43"></a> K., Pandey, N.: Increased prediction accuracy in the game of cricket using machine learning. arXiv preprint arXiv:1804.04226 (2018)
1. Basit,<a name="_page11_x129.58_y691.40"></a> A., Alvi, M.B., Jaskani, F.H., Alvi, M., Memon, K.H., Shah, R.A.: Icc t20 cricket world cup 2020 winner prediction using machine learning techniques. In: 2020 IEEE 23rd International Multitopic Conference (INMIC), pp. 1–6 (2020). IEEE
1. Singh,<a name="_page11_x129.58_y749.36"></a> T., Singla, V., Bhatia, P.: Score and winning prediction in cricket through

data mining. In: 2015 International Conference on Soft Computing Techniques and Implementations (ICSCTI), pp. 60–66 (2015). IEEE

6. Kumar,<a name="_page12_x105.05_y105.63"></a> J., Kumar, R., Kumar, P.: Outcome prediction of odi cricket matches using decision trees and mlp networks. In: 2018 First International Conference on Secure Cyber Computing and Communication (ICSCCC), pp. 343–347 (2018). IEEE
6. Pansare,<a name="_page12_x105.05_y163.59"></a> M.J., Khande, M.S., Oswal, A., Munsiff, Z., Choudhary, S., Kumbhar, V.: Cricket score prediction using xgboost regression
6. Souza,<a name="_page12_x105.05_y197.55"></a> F.M., Grando, J., Baldo, F.: Adaptive fast xgboost for regression. In: Intelligent Systems: 11th Brazilian Conference, BRACIS 2022, Campinas, Brazil, November 28–December 1, 2022, Proceedings, Part I, pp. 92–106 (2022). Springer
6. Roth,<a name="_page12_x105.05_y243.51"></a> V.: The generalized lasso. IEEE transactions on neural networks 15(1), 16–28 (2004)
6. Vovk,<a name="_page12_x100.07_y277.48"></a> V.: Kernel ridge regression. Empirical Inference: Festschrift in Honor of Vladimir N. Vapnik, 105–116 (2013)
6. Chicco,<a name="_page12_x100.07_y311.44"></a> D., Warrens, M.J., Jurman, G.: The coefficient of determination r-squared is more informative than smape, mae, mape, mse and rmse in regression analysis evaluation. PeerJ Computer Science 7, 623 (2021)
6. Bisong,<a name="_page12_x100.07_y357.40"></a> E., Bisong, E.: Matplotlib and seaborn. Building Machine Learning and Deep Learning Models on Google Cloud Platform: A Comprehensive Guide for Beginners, 151–165 (2019)
