from libraries import np, h5py
from helpers import gradient_checking, model

# array that has L elements which represents the number of units in each layer including the input layer

layers_dims = [12288, 20, 7, 5, 1]
activations = ["no activation", "relu", "relu", "relu", "sigmoid"]
L = len(layers_dims) - 1
# np.random.seed(1)


def main():
    train_x_orig, train_y, test_x_orig, test_y, _ = load_data(
        "train_catvnoncat.h5", "test_catvnoncat.h5"
    )
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(
        train_x_orig.shape[0], -1
    ).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.0
    test_x = test_x_flatten / 255.0

    parameters, costs, params_list, grads_list = model(
        train_x, train_y, activations, 1e-2, 2500
    )
    # Al = predict(test_x, test_y, parameters, activations)
    # _ = predict(train_x, train_y, parameters, activations)


def load_data(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File(test_path, "r")
    # your test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == "__main__":
    main()
