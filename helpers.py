import numpy as np

layers_dims = [12288, 20, 7, 5, 1]
activations = ["sigmoid", "relu", "relu", "relu", "sigmoid"]
L = len(layers_dims) - 1
epsilon = 1e-8
# np.random.seed(1)


def main():
    ...


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def der_relu(x):
    return (x > 0).astype(int)


def single_layer_initialization(units, n_x):
    Wi = np.random.randn(units, n_x)
    bi = np.zeros((units, 1))
    return Wi, bi


def network_initialization(layers_dims: list):
    L = len(layers_dims) - 1
    parameters = {}
    grads = {}
    for l in range(L):
        (
            parameters["W" + str(l + 1)],
            parameters["b" + str(l + 1)],
        ) = single_layer_initialization(layers_dims[l + 1], layers_dims[l])
        grads["dW" + str(l + 1)] = np.zeros((layers_dims[l + 1], layers_dims[l]))
        grads["db" + str(l + 1)] = np.zeros((layers_dims[l + 1], 1))
    return parameters, grads


def single_layer_forward_prop(Ai, Wi, bi, activation):
    Zi = np.dot(Wi, Ai) + bi  # shape = (no_of_units,m)
    if activation == "relu":
        Ai = relu(Zi)
    elif activation == "sigmoid":
        Ai = sigmoid(Zi)
    return Ai, Zi


def network_forward_prop(X: np.ndarray, parameters: dict, activation: list):
    A1, Z1 = single_layer_forward_prop(
        X, parameters["W1"], parameters["b1"], activation[1]
    )
    A_layer = A1
    cache = {}
    cache["Z1"] = Z1
    cache["A1"] = A1
    cache["A0"] = X
    for l in range(2, L + 1):
        A_layer, Z_layer = single_layer_forward_prop(
            A_layer,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation[l],
        )
        cache["A" + str(l)] = A_layer
        cache["Z" + str(l)] = Z_layer
    AL = A_layer
    return AL, cache


def compute_cost(Y: np.ndarray, AL: np.ndarray, type: str):
    if type == "cross entropy":
        cost = (
            -1
            * (
                (Y * np.log(np.add(np.abs(AL), epsilon)))
                + (1 - Y) * (np.log(np.add(np.abs(1 - AL), epsilon)))
            ).mean()
        )
        cost = np.squeeze(cost)
    return cost


def back_prop(
    AL: np.ndarray,
    Y: np.ndarray,
    parameters: dict,
    grads: dict,
    cache: dict,
    activations: list,
):
    dZ = AL - Y
    m = Y.shape[1]
    for l in range(L, 0, -1):
        grads["dW" + str(l)] = np.dot(dZ, (cache["A" + str(l - 1)]).T) / m
        grads["db" + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
        if l == 1:
            return grads
        dZ = np.dot(parameters["W" + str(l)].T, dZ)
        if activations[l - 1] == "sigmoid":
            dZ = dZ * der_sigmoid(cache["Z" + str(l - 1)])
        else:
            dZ = dZ * der_relu(cache["Z" + str(l - 1)])
    return grads


def update_parameters(parameters, learning_rate, grads):
    for l in range(L):
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        )
        return parameters


def dict_to_vector(dictionary: dict):
    vector = []
    indices = []
    for value in dictionary.values():
        indices.append(value.shape)
        vector.extend(value.flatten())
    return vector, indices


def vector_to_dict(vector: list, indices: list):
    mydict = {}
    last_index = 0
    for i, tup_index in enumerate(indices):
        index = tup_index[0] * tup_index[1] + last_index
        letter = "W" if i % 2 == 0 else "b"
        mydict[letter + str(i // 2 + 1)] = np.array(vector[last_index:index]).reshape(
            (tup_index[0], tup_index[1])
        )
        last_index = index
    return mydict


def gradient_checking(X, Y, parameters, grads, activations, error_type, eps=1e-7):
    vector, ind = dict_to_vector(parameters)
    grads, __ = dict_to_vector(grads)
    J_plus = np.zeros(len(vector))
    J_minus = np.zeros(len(vector))
    gradapprox = np.zeros(len(vector))
    for i in range(len(vector)):
        print(i)
        theta_plus = np.copy(vector)
        theta_plus[i] = vector[i] + eps
        Al, _ = network_forward_prop(X, vector_to_dict(theta_plus, ind), activations)
        J_plus[i] = compute_cost(Y, Al, error_type)

        theta_minus = np.copy(vector)
        theta_minus[i] = vector[i] - eps
        Al, _ = network_forward_prop(X, vector_to_dict(theta_minus, ind), activations)
        J_minus[i] = compute_cost(Y, Al, error_type)

        gradapprox[i] = J_plus[i] - J_minus[i] / (2 * eps)

    numerator = np.linalg.norm(grads - gradapprox)
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grads)
    diff = np.divide(numerator, denominator)
    if diff < 2 * eps:
        print("back prop works fine")
    else:
        print("there is a problem")
        print(diff)


def model(X, Y, activations, learning_rate, num_iterations):
    costs = []
    list_of_parameters = []
    list_of_grads = []
    parameters, grads = network_initialization(layers_dims)
    for i in range(num_iterations):
        AL, cache = network_forward_prop(X, parameters, activations)
        cost = compute_cost(Y, AL, "cross entropy")
        grads = back_prop(AL, Y, parameters, grads, cache, activations)
        list_of_parameters.append(parameters)
        list_of_grads.append(grads)
        parameters = update_parameters(parameters, learning_rate, grads)
        if (i % 100 == 0) or (i == num_iterations):
            costs.append(cost)
            print(f"Iteration: {i} cost: {cost}")
    return parameters, costs, list_of_parameters, list_of_grads


def predict(X: np.ndarray, Y: np.ndarray, parameters: dict, activations: list):
    AL, _ = network_forward_prop(X, parameters, activations)
    AL = np.round(AL, 0).astype(int)
    print("Accuracy: " + str(np.sum((AL == Y) / Y.shape[1])))
    return AL


if __name__ == "__main__":
    main()
