import tensorflow as tf
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Enable TensorFlow multi-threading
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

# Function to replace NaN values with the mean of each variable
def replace_nan_with_mean(data):
    nan_mean = np.nanmean(data)
    data[np.isnan(data)] = nan_mean
    return data

# Load the dataset
netcdf_file = r"/scratch/20cl91p02/ANN_BIO/BNN/ann_input_data.nc"
ds = xr.open_dataset(netcdf_file)

# Extract data variables
fe = ds['fe'].values  # (time, depth, lat, lon)
po4 = ds['po4'].values
si = ds['si'].values
no3 = ds['no3'].values  # Predictor
nppv = ds['nppv'].values  # Target variable
latitude = ds['latitude'].values
longitude = ds['longitude'].values

# Discard the depth dimension and focus on (time, lat, lon)
fe = fe[:, 0, :, :]
po4 = po4[:, 0, :, :]
si = si[:, 0, :, :]
no3 = no3[:, 0, :, :]
nppv = nppv[:, 0, :, :]  # Ensure this matches the structure

# Replace NaN values in the input and target data
fe = replace_nan_with_mean(fe)
po4 = replace_nan_with_mean(po4)
si = replace_nan_with_mean(si)
no3 = replace_nan_with_mean(no3)
nppv = replace_nan_with_mean(nppv)

# Stack the input variables along a new channel dimension
inputs = np.stack([fe, po4, si, no3], axis=-1)

# Prepare input for LSTM
time_steps = 5
samples = inputs.shape[0] - time_steps
X_lstm = np.array([inputs[i:i + time_steps] for i in range(samples)])
y_lstm = nppv[time_steps:]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[2] * X_train.shape[3] * X_train.shape[4])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2] * X_test.shape[3] * X_test.shape[4])
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

scaler_y = StandardScaler()
y_train_reshaped = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])
y_test_reshaped = y_test.reshape(-1, y_test.shape[1] * y_test.shape[2])
y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

# Define the Bayesian Neural Network (BNN) + LSTM Model
class BNNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', dropout_rate=0.2):
        super(BNNLayer, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dense = tf.keras.layers.Dense(units, activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        if training:
            x = self.dropout(x, training=training)
        if self.activation:
            x = tf.keras.activations.get(self.activation)(x)
        return x

inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))(inputs)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
x = tf.keras.layers.LSTM(64, activation='relu')(x)
x = BNNLayer(64)(x)
outputs = tf.keras.layers.Dense(np.prod(y_train_scaled.shape[1:]))(x)
outputs = tf.keras.layers.Reshape(y_train_scaled.shape[1:])(outputs)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test_scaled)

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_y = scaler_y.inverse_transform(predictions.reshape(-1, y_test.shape[1] * y_test.shape[2])).reshape(y_test.shape)

# Compute average actual and predicted values across time steps
average_actual_nppv = np.nanmean(y_test, axis=0)
average_predicted_nppv = np.nanmean(predicted_y, axis=0)

# Save output to NetCDF
output_file_path = r"/scratch/20cl91p02/ANN_BIO/BNN/average_output_bnn_lstm_nppv.nc"
ds_out = xr.Dataset(
    data_vars={
        "predicted_nppv": (("latitude", "longitude"), average_predicted_nppv),
        "actual_nppv": (("latitude", "longitude"), average_actual_nppv),
    },
    coords={"latitude": latitude, "longitude": longitude},
)
ds_out.to_netcdf(output_file_path)
