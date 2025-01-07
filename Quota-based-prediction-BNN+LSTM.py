import tensorflow as tf
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to replace NaN values with the mean of each variable
def replace_nan_with_mean(data):
    # Compute the mean along the entire array, ignoring NaNs
    nan_mean = np.nanmean(data)  # This will compute a single mean value for all NaN entries
    # Replace NaNs with the computed mean
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

# Extract latitude and longitude
latitude = ds['latitude'].values  # Shape: (lat,)
longitude = ds['longitude'].values  # Shape: (lon,)

# Since depth is constant, discard the depth dimension and focus on (time, lat, lon)
fe = fe[:, 0, :, :]
po4 = po4[:, 0, :, :]
si = si[:, 0, :, :]
no3 = no3[:, 0, :, :]
nppv = nppv[:, 0, :, :]  # Ensure this matches the structure

# Replace NaN values in the input data (fe, po4, si, no3) and the target data (nppv)
fe = replace_nan_with_mean(fe)
po4 = replace_nan_with_mean(po4)
si = replace_nan_with_mean(si)
no3 = replace_nan_with_mean(no3)
nppv = replace_nan_with_mean(nppv)

# Stack the input variables along a new channel dimension (fe, po4, si, no3)
inputs = np.stack([fe, po4, si, no3], axis=-1)  # Shape: (time, lat, lon, channels)

# Prepare input for LSTM
time_steps = 5  # Number of time steps to consider in each sequence
samples = inputs.shape[0] - time_steps
X_lstm = np.array([inputs[i:i + time_steps] for i in range(samples)])

# Target: Predict NPPV
y_lstm = nppv[time_steps:]  # Shape: (samples, lat, lon)

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

# Define a custom BNN layer using Monte Carlo (MC) dropout
class BNNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', dropout_rate=0.2):
        super(BNNLayer, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dense = tf.keras.layers.Dense(units, activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        # This method is called when the layer is built.
        # You can access input_shape and initialize layer weights if needed.
        pass

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        if training:
            x = self.dropout(x, training=training)
        if self.activation:
            x = tf.keras.activations.get(self.activation)(x)
        return x

    def compute_output_shape(self, input_shape):
        # Assuming the output of the Dense layer is (batch_size, units)
        return (input_shape[0], self.units)


# Define the model with BNN + LSTM
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))  # Adjust for 5D input
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))(inputs)  # Apply Conv2D in time-distributed manner

# Flatten the spatial dimensions (lat * lon) and channels (4) while keeping the time dimension intact
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

# Now x should have shape (samples, time_steps, flattened_features)
x = tf.keras.layers.LSTM(64, activation='relu')(x)  # LSTM layer to process time steps
x = BNNLayer(64)(x)  # BNN layer (custom layer, if you have one)

# Define output shape to match the target (lat, lon)
outputs = tf.keras.layers.Dense(np.prod(y_train_scaled.shape[1:]))(x)
outputs = tf.keras.layers.Reshape(y_train_scaled.shape[1:])(outputs)  # Reshape to (lat, lon)

# Model definition
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled,
                    epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test_scaled)

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_y = scaler_y.inverse_transform(predictions.reshape(-1, y_test.shape[1])).reshape(y_test.shape)

print("Test Loss:", test_loss)

# Compute average actual and predicted values across time steps
average_actual_nppv = np.nanmean(y_test, axis=0)  # Average across time dimension
average_predicted_nppv = np.nanmean(predicted_y, axis=0)

# Define output file path
output_file_path = r"/scratch/20cl91p02/ANN_BIO/BNN/average_output_bnn_lstm_nppv.nc"

# Create a new NetCDF file
with xr.open_dataset(output_file_path, mode='w') as ds_out:
    # Create dimensions
    ds_out.coords['latitude'] = ('latitude', latitude)
    ds_out.coords['longitude'] = ('longitude', longitude)
    
    # Create the variable for predicted NPPV and actual NPPV
    ds_out.createVariable('predicted_nppv', ('latitude', 'longitude'), data=average_predicted_nppv)
    ds_out.createVariable('actual_nppv', ('latitude', 'longitude'), data=average_actual_nppv)
    
    # Save to NetCDF
    ds_out.to_netcdf(output_file_path)
