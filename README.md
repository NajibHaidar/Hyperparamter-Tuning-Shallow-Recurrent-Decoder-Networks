# Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks

### Table of Contents
[Abstract](#Abstract)
<a name="Abstract"/>

[Sec. I. Introduction and Overview](#sec-i-introduction-and-overview)     
<a name="sec-i-introduction-and-overview"/>

[Sec. II. Theoretical Background](#sec-ii-theoretical-background)     
<a name="sec-ii-theoretical-background"/>

[Sec. III. Algorithm Implementation and Development](#sec-iii-algorithm-implementation-and-development)
<a name="sec-iii-algorithm-implementation-and-development"/>

[Sec. IV. Computational Results](#sec-iv-computational-results)
<a name="sec-iv-computational-results"/>

[Sec. V. Summary and Conclusions](#sec-v-summary-and-conclusions)
<a name="sec-v-summary-and-conclusions"/>

### Abstract

In this study, we explore the impact of time lag, noise, and the number of sensors on the performance of a Long Short-Term Memory (LSTM) based decoder model in the context of sea-surface temperature prediction. The model is trained on sea-surface temperature data, with the objective of understanding the underlying temporal and spatial patterns that drive these temperatures. The performance of the model is then examined under varying conditions such as different time lags, addition of Gaussian noise to the data, and changing the number of sensors in the input data. The insights gained from this study can be instrumental in understanding the intricacies of applying LSTM models to environmental prediction tasks, and the factors that can significantly influence their performance.

### Introduction and Overview

**Introduction:**

The prediction of sea-surface temperature is a significant task in environmental science and meteorology. Given the temporal and spatial complexity of sea-surface temperature patterns, machine learning models, specifically those capable of handling sequential data such as LSTM, have emerged as powerful tools to tackle this problem.

However, the performance of these models can be influenced by several factors, including the time lag between observations, the level of noise in the data, and the number of sensors used to capture the measurements. Understanding the impact of these factors on model performance is crucial for optimizing these models and ensuring accurate and reliable predictions.

In this study, we train an LSTM-based model on sea-surface temperature data, and systematically vary the time lag, noise, and the number of sensors to analyze their impact on the model's performance. The aim is to understand how these factors influence the model's ability to capture and predict sea-surface temperature patterns, and to provide insights that can help in the design and application of LSTM models for similar tasks in the future.

**Overview:**

The remainder of this study is organized as follows:

First, we present the LSTM-based model used in this study, and detail the procedure for training the model on the sea-surface temperature data. This includes an explanation of how the time lag, noise, and the number of sensors are incorporated into the model and the training process.

Next, we present the results of our experiments. For each of the factors under consideration - time lag, noise, and the number of sensors - we present an analysis of how changes in that factor influence the model's performance. This includes both quantitative results, such as changes in prediction accuracy, and qualitative observations about how the model's predictions change.

Finally, we conclude with a summary of our findings, highlighting the key insights gained about the impact of time lag, noise, and the number of sensors on the performance of LSTM models in the context of sea-surface temperature prediction. We also discuss potential implications of these findings for future work in this area.

### Theoretical Background
**Long Short-Term Memory (LSTM) Models:**

Long Short-Term Memory (LSTM) models are a special kind of Recurrent Neural Network (RNN) that were introduced by Hochreiter and Schmidhuber in 1997. Unlike traditional RNNs, LSTMs have an improved ability to capture long-term dependencies in sequence data, which makes them more suitable for tasks involving data where temporal dynamics play a critical role, such as in our case of sea-surface temperature prediction.

LSTMs achieve this through a unique architectural element known as a 'cell state', which runs through the entire sequence, and gates that control the flow of information into and out of the cell state. These gates - the forget gate, input gate, and output gate - allow the model to selectively remember or forget information, which helps in capturing temporal dependencies over long periods.

**Sea-Surface Temperature Prediction:**

Sea-surface temperature prediction is a complex task due to the multitude of factors that influence sea-surface temperatures. These include atmospheric conditions, ocean currents, solar radiation, and more. Machine learning models, particularly those capable of handling sequence data like LSTM, have shown promise in capturing these complex dynamics and providing accurate predictions.

However, the performance of these models can be influenced by several factors. One such factor is the time lag between observations, which determines how far back in time the model looks to make a prediction. A larger time lag may allow the model to capture longer-term patterns in the data, but it also increases the complexity of the model and the amount of data required for training.

**Noise and Model Performance:**

In real-world scenarios, data collected from sensors often comes with some level of noise. Noise can be thought of as random or unpredictable fluctuations in the data that do not reflect the underlying pattern or trend. When training a model, it is important to understand how the presence of noise in the data can impact the model's performance.

High levels of noise can make it difficult for the model to discern the underlying patterns in the data, leading to poorer performance. On the other hand, a model trained on noisy data may be more robust to noise in future data it encounters. In this study, we introduce Gaussian noise to the sea-surface temperature data to investigate its impact on the performance of the LSTM model.

**Number of Sensors and Model Performance:**

The number of sensors used to collect data can also significantly impact the performance of a model. More sensors can provide a richer, more detailed view of the environment, potentially allowing the model to capture more complex patterns. However, more sensors also mean more data, which can increase the complexity of the model and the computational resources required for training and inference.

In our study, we vary the number of sensors used to collect the sea-surface temperature data and analyze its impact on the LSTM model's performance. This can provide insights into the trade-off between the number of sensors and model performance, and help in determining an optimal number of sensors for this task.

**Model Evaluation:**

In this study, we evaluate the performance of the LSTM model using a commonly used metric in regression tasks - the Mean Squared Error (MSE). The MSE measures the average squared difference between the model's predictions and the actual values, providing an indication of the model's accuracy. Lower MSE values indicate better performance.

It's important to note that in addition to quantitative metrics like MSE, qualitative analysis of the model's predictions can also provide useful insights. For instance, visualizing the model's predictions over time can reveal whether the model is able to capture the temporal dynamics of sea-surface temperature.

### Algorithm Implementation and Development

The run_experiment function forms the core of the modeling and evaluation process. Here's a comprehensive explanation of the steps involved in this function:

```
def run_experiment(num_sensors, lags, num_epochs, noise_std):
```

The function takes four parameters: num_sensors, which is the number of sensors in use, lags, the number of lagging time steps, num_epochs, the number of iterations over the dataset during training, and noise_std, the standard deviation of the Gaussian noise added to the input data.

```
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```

The load_data function is used to load the dataset, 'SST'. The shape of the data is recorded in n and m respectively. Then, sensor locations are randomly chosen from the available locations without replacement.

```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```

A training set of 1000 samples is randomly selected, with a mask applied to ensure these samples are not used for validation or testing. The remaining data is split into validation and test sets by taking alternate indices.

```
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)
```

The data is scaled using the MinMaxScaler, which scales the data to a specified range, typically between zero and one. It fits the scaler on the training data and transforms the whole dataset.

The following block of code generates input sequences for a SHRED model and adds Gaussian noise to the input data:

```
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

# Add Gaussian noise to the input data
noise = np.random.normal(0, noise_std, all_data_in.shape)
all_data_in += noise
```

A tensor of zeros is initialized with the shape (n - lags, lags, num_sensors) to store the input sequences. Each sequence consists of lagged values for randomly selected sensors from the scaled data. Gaussian noise, with mean zero and standard deviation specified by noise_std, is then added to these sequences.

```
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

This line determines whether a CUDA-capable GPU is available for computation. If so, it sets the device to 'cuda', otherwise it uses the CPU ('cpu').

```
train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)
```

These lines create PyTorch tensorsquote("The run_experiment function forms", "PyTorch tensors") from the input data for training, validation, and testing. They are moved to the appropriate device (CPU or GPU) for computation.

```
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)
```

Similarly, tensors for the target values (corresponding to the last timestep of each sequence) are created for training, validation, and testing.

```
train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```

The input and target tensors are wrapped into PyTorch Dataset objects for efficient data loading and batching.

```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
```

A SHRED model is initialized with the given parameters and moved to the appropriate device for computation.

```
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=num_epochs, lr=1e-3, verbose=True, patience=5)
```

The SHRED model is trained on the training dataset using the specified number of epochs, learning rate, and other parameters. The model's performance is evaluated on the validation dataset at each epoch, and training stops early if the validation performance does not improve for a specified number of epochs ('patience').

```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
test_performance = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
print(test_performance)
```

Finally, the trained SHRED model is used to make predictions on the test dataset. The predictions are inverse-transformed back to the original scale, and the model's performance is evaluated by comparing these predictions to the ground truth data. The performance metric used here is the relative Euclidean norm of the prediction error.

```
return shred, validation_errors, test_performance
```

The function returns the trained SHRED model, the history of validation errors during training, and the final test performance.

This comprehensive run_experiment function encapsulates the process of preparing the data, setting up the model, training the model, and evaluating its performance. By abstracting these steps into a single function, it enables streamlined and consistent experimentation with different parameters and configurations.


```
num_sensors = 3
lags = 52
num_epochs = 1000
noise_std = 0

shred_OG, validation_errors_OG, test_performance_OG = run_experiment(num_sensors, lags, num_epochs, noise_std)
```

In this block of code, the run_experiment function is called with these parameters to train a SHRED model. The function returns a trained model (shred_OG), the validation errors at each epoch (validation_errors_OG), and the final performance of the model on the test set (test_performance_OG).

```
import pandas as pd

def parse_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if "Training epoch" in line:
            epoch = int(line.split()[-1])
        elif "Error" in line:
            error = float(line.split()[-1].replace("tensor(", "").replace(")", ""))
            data.append({'epoch': epoch, 'error': error})

    return pd.DataFrame(data)

df = parse_file("training_result.txt")
```

Next, a helper function parse_file is defined to parse the output of the training process from a text file. This function reads the file line by line, and for each line that contains "Training epoch", it extracts the epoch number. For each line that contains "Error", it extracts the error value. Finally, this function is used to parse the file "training_result.txt" into a pandas DataFrame for further analysis.

This DataFrame df now contains a record of the training process, with each row representing an epoch and columns for the epoch number and the error at that epoch. This data can be used for various purposes, such as tracking the training progress, diagnosing issues with the training process, and visualizing the training and validation errors over time. This method was only used for the experiment with unchanging variables in order to determine where the error seems to fade. In other experiments, we used dictionaries instead as we will see shortly.

![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/77052126-75dd-4900-8853-08dbf4915735)
*Figure 1: Fit of Non-Sweeping Parameters over 1000 Epochs*

From figure 1, we deduced that we could save plenty of training time by only triaining the data until 200 epochs (1/5th the original 1000 epochs). This is possible because it is evident that the error seems to barely change from that point forward. 

Then, we conducted a series of experiments with the SHRED model, varying the amount of lag in the input data. The lags_range is set to a list of values ranging from 0 to 260. These values represent the number of previous time steps to be considered while making a prediction in the model. The number of sensors (num_sensors), the number of training epochs (num_epochs), and the standard deviation of the Gaussian noise (noise_std) added to the input data are set as constants for all experiments.

```
num_sensors = 3
lags_range = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 75, 104, 156, 208, 260]
num_epochs = 200
noise_std = 0
```

Two dictionaries, validation_errors_dict_lag and test_performance_dict_lag, are initialized to store the validation errors and test performance, respectively, for each value of lag. Each key-value pair in these dictionaries corresponds to a specific experiment, with the key being the amount of lag and the value being the validation errors or test performance.

```
validation_errors_dict_lag = {}
test_performance_dict_lag = {}
```

A loop is then used to run an experiment for each value of lag in lags_range. In each iteration of the loop, the run_experiment function is called with the current value of lag, and the resulting SHRED model, validation errors, and test performance are stored.

```
for lags in lags_range:
    print(f"Running experiment with {lags} lag...")
    shred_lag, validation_errors_lag, test_performance_lag = run_experiment(num_sensors, lags, num_epochs, noise_std)

    # Store validation errors and test performance
    validation_errors_dict_lag[lags] = validation_errors_lag
    test_performance_dict_lag[lags] = test_performance_lag
```

After running this code, validation_errors_dict_lag and test_performance_dict_lag will contain the results of all experiments, which can be used for further analysis, such as identifying the optimal amount of lag for the SHRED model.


This same exact procedure was then repeated for varying values of num_sensors and noise_std. Of course, each case had its own dedicated dictionary:

```
num_sensors_range = range(1, 11)  # change this to the range you want
lags = 52
num_epochs = 200
noise_std = 0

# Dictionaries to store validation errors and test performance for each experiment
validation_errors_dict_sensor = {}
test_performance_dict_sensor = {}

for num_sensors in num_sensors_range:
    print(f"Running experiment with {num_sensors} sensors...")
    shred_sensor, validation_errors_sensor, test_performance_sensor = run_experiment(num_sensors, lags, num_epochs, noise_std)

    # Store validation errors and test performance
    validation_errors_dict_sensor[num_sensors] = validation_errors_sensor
    test_performance_dict_sensor[num_sensors] = test_performance_sensor
```

```
noise_range = range(0, 11)  # change this to the range you want
num_sensors = 3
lags = 52
num_epochs = 200

# Dictionaries to store validation errors and test performance for each experiment
validation_errors_dict_noise = {}
test_performance_dict_noise = {}

for noise_std in noise_range:
    print(f"Running experiment with noise standard deviation {noise_std}...")
    shred_noise, validation_errors_noise, test_performance_noise = run_experiment(num_sensors, lags, num_epochs, noise_std)

    # Store validation errors and test performance
    validation_errors_dict_noise[noise_std] = validation_errors_noise
    test_performance_dict_noise[noise_std] = test_performance_noise
```

### Computational Results
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/3dc3a136-2141-466a-bbc8-cc014eb76997)
*Figure 2: Fit of Non-Sweeping Parameters over 1000 Epochs*

![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/66a1960e-27e9-4cc4-8b90-a78ce2aea0c2)
*Figure 3: Fit of Non-Sweeping Parameters over 1000 Epochs*

Anyalzing the sensor data in figures 2 and 3, we can observe that the test performance (which represents the error of the model on unseen data) generally decreases as the number of sensors increases, indicating that the model's performance improves with more sensors. This trend suggests that using more sensors allows the model to capture more relevant features, thereby improving its predictive accuracy.

However, it's important to note that the improvement in performance tends to diminish as the number of sensors increases. For instance, the decrease in test performance from 1 sensor to 2 sensors is larger than the decrease from 9 sensors to 10 sensors. This pattern suggests that there may be diminishing returns to adding more sensors.

We can also look at the validation errors for each model. The validation error is a measure of the model's performance on a held-out subset of the training data. By looking at the validation errors, we can get an idea of how well the model is likely to perform on unseen data.

We can see that the minimum validation error generally decreases as the number of sensors increases. This trend is consistent with the trend in the test performance, further supporting the conclusion that the model's performance improves with more sensors.

However, the spread of validation errors seems to increase with the number of sensors. This spread may suggest that the choice of hyperparameters becomes more important as the number of sensors increases. In other words, with more sensors, there may be a greater risk of overfitting the model to the training data, which would result in poor performance on unseen data. Therefore, careful tuning of the model's hyperparameters is likely to be especially important when using a large number of sensors.

![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/d13ec847-25d6-45e4-9a70-d59f9fe7ac5e)
*Figure 4: Fit of Non-Sweeping Parameters over 1000 Epochs*

![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/ee37d0c7-52b5-40e3-af7f-128bc12896ae)
*Figure 5: Fit of Non-Sweeping Parameters over 1000 Epochs*

From the lag data in figures 4 and 5, we can see that when the amount of lag is 0, the model cannot produce any valid outputs, resulting in NaN errors. This is likely because the LSTM model needs at least some lag to learn temporal dependencies in the data, and with zero lag, it essentially becomes a simple feedforward neural network. 

Starting from a lag of 1, the model is able to learn and make predictions, but the test performance is relatively poor. As the amount of lag increases, the test performance generally improves, which suggests that more lag allows the model to capture longer-term temporal dependencies in the data, improving its ability to make predictions.

However, the trend isn't perfectly smooth. The test performance seems to reach a minimum around a lag of 15, then increases slightly before decreasing again around a lag of 40-50, and then finally increases again. These variations could be due to random fluctuations in the training process, or they could suggest that there's a complex relationship between the amount of lag and the model's performance. 

For example, it's possible that a moderate amount of lag allows the model to capture the relevant temporal dependencies without overwhelming it with too much information, while a large amount of lag could introduce additional complexity that the model struggles to handle. On the other hand, too little lag might not provide enough temporal context for the model to make accurate predictions.

Looking at the validation errors, we can see a similar trend. The minimum validation error generally decreases as the amount of lag increases, suggesting that more lag improves the model's performance on the validation set. However, the spread of validation errors also seems to increase with more lag, suggesting that the choice of hyperparameters becomes more important as the amount of lag increases.

Overall, these results suggest that there's a trade-off when choosing the amount of lag for the LSTM model. Too little lag might not provide enough temporal context for the model to make accurate predictions, but too much lag might introduce additional complexity that the model struggles to handle. Therefore, the optimal amount of lag likely depends on the specific characteristics of the data and the task. 

In this case, given the noise level and the number of sensors used, a lag in the range of 15 to 50 seems to result in the best performance. Beyond this range, the performance does not seem to improve significantly and in some cases even deteriorates slightly. It would be important to note that this optimum lag range might change if other parameters such as the number of sensors or the noise level are altered. 

It would also be worth exploring whether different preprocessing or feature engineering techniques could help the model handle larger amounts of lag more effectively. For example, it could be beneficial to explore techniques for dimensionality reduction, sequence compression, or more advanced techniques for handling temporal data. 

As always, it's important to keep in mind that these results are based on a specific model and a specific dataset, so the generalizability of these conclusions to other models or datasets may be limited.

![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/ca1dea00-dbda-4bde-a354-6565ab132b30)
*Figure 6: Fit of Non-Sweeping Parameters over 1000 Epochs*

![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/e06cf762-3335-4eb8-80c2-2e1af61fefe7)
*Figure 7: Fit of Non-Sweeping Parameters over 1000 Epochs*

The effect of noise standard deviation on the LSTM model's performance is shown in the provided data fugures 6 and 7 . The noise standard deviation ranges from 0 to 10.

When the noise standard deviation is 0 (i.e., no noise), the test performance is 0.0292. As the noise standard deviation increases from 1 to 10, the test performance generally increases, indicating that the model's prediction performance degrades as the noise level in the data increases.

At a noise standard deviation of 1, the test performance is 0.0652, and at a standard deviation of 2, the test performance further increases to 0.0982. This pattern continues up to a noise standard deviation of 10, at which point the test performance is 0.1404.

This trend shows that the LSTM model's performance in predicting sea-surface temperature declines as the noise level in the data increases. This is expected because noise introduces randomness and uncertainty into the data, which makes it harder for the model to learn the underlying patterns in the data. This finding suggests that it's important to ensure the quality of the input data and minimize noise when using LSTM models for environmental prediction tasks. 

Also, please note that the validation errors are given as tensors for noise standard deviations greater than 0. This might be due to an issue with the formatting or processing of the data. It would be more helpful to have these errors in the same format as the rest of the data to make accurate comparisons.


### Summary and Conclusions
