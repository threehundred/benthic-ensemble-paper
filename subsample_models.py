# import libraries

import pickle
import numpy as np
import random

def reshape_mc_predictions(X):
    # change the shape from N * M to M * N

    X = np.array(X)
    new_list = []
    for index in range(0, len(X[0])):
        new_row = X[:, index, None]
        new_list.append(new_row)

    # print("new len", len(new_list), len(new_list[0]))
    return new_list


def generate_vanilla(inference_run):
    index = random.randint(0, 14)

    X_path = inference_run["X_path"]
    X = pickle.load(open(X_path, "rb"))

    vanilla_X = [X[index]]

    return vanilla_X


def generate_vanillas(inference_runs, string_prefix, base_path):
    # random select runs
    run_indexes = random.choices(range(0, 14), k=10)
    inference_runs = np.take(inference_runs, run_indexes, 0)

    for i, inference_run in enumerate(inference_runs):
        new_X = generate_vanilla(inference_run)

        # save new X & y
        x_path = "%s%s%s%s" % (base_path, string_prefix, str(i + 1), ".p")
        pickle.dump(new_X, open(x_path, "wb"))

        y_path = inference_run["y_val_path"]
        y = pickle.load(open(y_path, "rb"))
        new_y_path = "%s%s%s%s%s" % (base_path, string_prefix, "y", str(i + 1), ".p")
        pickle.dump(y, open(new_y_path, "wb"))


def generate_bayes(inference_run):
    indexes = random.choices(range(0, 14), k=10)

    X_path = inference_run["X_path"]
    X = pickle.load(open(X_path, "rb"))

    bayes_X = np.take(X, indexes, 0)

    return bayes_X


def generate_bayesians(inference_runs, string_prefix, base_path):
    # random select runs
    run_indexes = random.choices(range(0, 14), k=10)
    inference_runs = np.take(inference_runs, run_indexes, 0)

    for i, inference_run in enumerate(inference_runs):
        new_X = generate_bayes(inference_run)

        # save new X & y
        x_path = "%s%s%s%s" % (base_path, string_prefix, str(i + 1), ".p")
        pickle.dump(new_X, open(x_path, "wb"))

        y_path = inference_run["y_val_path"]
        y = pickle.load(open(y_path, "rb"))
        new_y_path = "%s%s%s%s%s" % (base_path, string_prefix, "y", str(i + 1), ".p")
        pickle.dump(y, open(new_y_path, "wb"))


def generate_ensemble(inference_runs, no_of_ensembles=10):
    # random select runs
    run_indexes = random.choices(range(0, 14), k=no_of_ensembles)
    inference_runs = np.take(inference_runs, run_indexes, 0)

    new_X = []

    # load the X's
    for i, inference_run in enumerate(inference_runs):
        # load X
        X_path = inference_run["X_path"]
        X = pickle.load(open(X_path, "rb"))

        # random select from X
        index = random.randint(0, 14)
        new_X.append(X[index])

    return new_X


def generate_ensembles(inference_runs, string_prefix, base_path):
    # random select runs
    # run_indexes = random.choices(range(0, 14), k=10)
    # inference_runs = np.take(inference_runs, run_indexes, 0)

    for i in range(0, 10):
        new_X = generate_ensemble(inference_runs)

        # save new X & y
        x_path = "%s%s%s%s" % (base_path, string_prefix, str(i + 1), ".p")
        pickle.dump(new_X, open(x_path, "wb"))

        y_path = inference_runs[0]["y_val_path"]
        y = pickle.load(open(y_path, "rb"))
        new_y_path = "%s%s%s%s%s" % (base_path, string_prefix, "y", str(i + 1), ".p")
        pickle.dump(y, open(new_y_path, "wb"))


if __name__ == "__main__":

    # do the validation data
    pickle_paths = []

    for i in range(1, 16):
        pickle_paths.append(
            {
                "X_path": "./inferenceoutputs/Trip1-inference-test-GROUP_DESC-%s.p" % str(i),
                "y_val_path": "./inferenceoutputs/Trip1-inferencey-test-GROUP_DESC-%s.p" % str(i),
            }
        )

    generate_vanillas(pickle_paths, "Trip1-test-GROUP_DESC-vanilla-inference", "./data_DESC2_ensem/")
    generate_bayesians(pickle_paths, "Trip1-test-GROUP_DESC-bayes-inference", "./data_DESC2_ensem/")
    generate_ensembles(pickle_paths, "Trip1-test-GROUP_DESC-ensemble-inference", "./data_DESC2_ensem/")

    # do the blur data
    for blur in range(1, 11):
        pickle_paths = []

        for i in range(1, 16):
            pickle_paths.append(
                {
                    "X_path": "./inferenceoutputs/Trip1-inference-test-GROUP_DESC-blur%s-%s.p" % (str(blur), str(i)),
                    "y_val_path": "./inferenceoutputs/Trip1-inference-test-GROUP_DESC-blur%s-%s.p" % (str(blur), str(i)),
                }
            )

        generate_vanillas(pickle_paths, "Trip1-test-GROUP_DESC-vanilla-inference-blur%s" % str(blur), "./data_DESC2_ensem/")
        generate_bayesians(pickle_paths, "Trip1-test-GROUP_DESC-bayes-inference-blur%s" % str(blur), "./data_DESC2_ensem/")
        generate_ensembles(pickle_paths, "Trip1-test-GROUP_DESC-ensemble-inference-blur%s" % str(blur), "./data_DESC2_ensem/")

    # do the colour data
    for col in range(1, 11):
        pickle_paths = []

        for i in range(1, 16):
            pickle_paths.append(
                {
                    "X_path": "./inferenceoutputs/Trip1-inference-test-GROUP_DESC-colour%s-%s.p" % (str(col), str(i)),
                    "y_val_path": "./inferenceoutputs/Trip1-inference-test-GROUP_DESC-colour%s-%s.p" % (str(col), str(i)),
                }
            )

        generate_vanillas(pickle_paths, "Trip1-test-GROUP_DESC-vanilla-inference-colour%s" % str(col), "./data_DESC2_ensem/")
        generate_bayesians(pickle_paths, "Trip1-test-GROUP_DESC-bayes-inference-colour%s" % str(col), "./data_DESC2_ensem/")
        generate_ensembles(pickle_paths, "Trip1-test-GROUP_DESC-ensemble-inference-colour%s" % str(col), "./data_DESC2_ensem/")