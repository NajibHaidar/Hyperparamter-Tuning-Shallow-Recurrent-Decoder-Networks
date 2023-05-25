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

### Computational Results
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/3dc3a136-2141-466a-bbc8-cc014eb76997)
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/66a1960e-27e9-4cc4-8b90-a78ce2aea0c2)
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/d13ec847-25d6-45e4-9a70-d59f9fe7ac5e)
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/ee37d0c7-52b5-40e3-af7f-128bc12896ae)
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/ca1dea00-dbda-4bde-a354-6565ab132b30)
![image](https://github.com/NajibHaidar/Hyperparamter-Tuning-Shallow-Recurrent-Decoder-Networks/assets/116219100/e06cf762-3335-4eb8-80c2-2e1af61fefe7)




### Summary and Conclusions
