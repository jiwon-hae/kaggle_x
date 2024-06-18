from data.data_io import *
from data.data_processing import standardize_data
from model.regression import *

if __name__ == '__main__':
    x_train, y_train = load_data('train')
    x_test, _ = load_data('test')

    x_train, x_val, y_train, y_val = split_data(x_train, y_train)
    x_train, x_val = standardize_data(x_train, x_val)

    X_train = data_frame_to_tensor(x_train)
    y_train = data_frame_to_tensor(y_train.values).view(-1, 1)
    X_val = data_frame_to_tensor(x_val)
    y_val = data_frame_to_tensor(y_val.values).view(-1, 1)
    # X_test = data_frame_to_tensor(x_test)

    input_dim = X_train.shape[1]
    model = LinearRegressionModel(input_dim)
    model.fit(num_epochs=100000, x_train=X_train, y_train=y_train)
    # model.evaluate(x_train, y_train, x_val, y_val)
