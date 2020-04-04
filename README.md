 # LSTM Stock Predictor
 #### Deep Learning
 ---

In this assignment, I use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.

### Evaluating the performance of each model
 After experimenting with different window sizes, and trying different values to train both models, I made the following conclusions:
* A window size of one led to better results for both models
* Training the models with higher values for both epochs and batch sizes resulted in a larger loss during testing. 
* Using the `Fear and Greed Index' as a feauture to train the model resulted in a bigger loss percentage compared to the second model. The lowest loss I achieved was 9.46% for the first model. Using the same parameter values to train the next model resulted in a loss percentage of 0.17%
* The second model, using closing prices, performs better in tracking actual stock prices over time. 




