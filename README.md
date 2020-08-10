 # Build and Evaluate Deep Learning Models
 ## Overview
 In this project we will develop and evaluate two custom `Long Short-Term Memory Recurrent Neural Network` models that predict Bitcoin `nth` day closing price based on a rolling `X` day window. The first model will use the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/), which analyzes emotions and sentiments from different sources to produce a daily FNG value for cryptocurrencies. The second model will use Bitcoin closing prices. During training we will experiment with different values for the following parameters: `window size(lookback window), number of input layers, epochs, and batch size`. Each model will be evaluated on test data(unseen data). This process can be repeated multiple times until we find a model with the best performace. Then use the model to make predictions and compare them to actual values.

 Note: In order to make accurate comparisons between the two models, we need to maintain the same architecture and parameters during training and testing of each model. 

### Evaluating the performance of each model
 After experimenting with different window sizes, and trying different values to train both models, I made the following conclusions:
* A window size of one led to better results for both models
* Training the models with higher values for both epochs and batch sizes resulted in a larger loss during testing. 
* Using the `Fear and Greed Index' as a feauture to train the model resulted in a bigger loss percentage compared to the second model. The lowest loss I achieved was 9.46% for the first model. Using the same parameter values to train the next model resulted in a loss percentage of 0.17%
* The second model, using closing prices, performs better in tracking actual stock prices over time. 




