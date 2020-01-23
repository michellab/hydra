# General:
import pandas as pd
import numpy as np
import os
import sys
import csv
import time
import shutil
import logging

# Tensorflow:
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 3"  # current workstation contains 4 GPUs; exclude 1st

# Sklearn
from skopt import gp_minimize, dump
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args

# Path variables:
path = './'
datasets_dr = '../datasets/'
SDF_dr = datasets_dr + 'sdffiles/'
output_dr = path + 'output/'
if not os.path.exists(output_dr):
    os.mkdir(output_dr)
figures_dr = path + 'figures/'
if not os.path.exists(output_dr):
    os.mkdir(figures_dr)

# Global variables:
model_type = 'DNN'
offset_col_name = 'dGoffset (kcal/mol)'

# DNN global variables
n_calls = 20  # Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
epochs = 200
best_mae = np.inf

# load in data set
dtrain_df = pd.read_hdf(datasets_dr + 'dtrain_data.h5', key='relative')
num_input_nodes = len(dtrain_df.columns) - 1


def main():

    # initiate log file
    logging.basicConfig(filename= output_dr + 'training_logfile.txt',
                    filemode='a',
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO)
    logging.info('Starting {}.'.format(__file__))
    logging.info('\n\nDNN parameters:\n\nn_calls = {}  # gp_minimize\nepochs = {}\n'.format(n_calls, epochs))

    train_model(dtrain_df)

    logging.info('Finished dGhydr_train_{}.py.'.format(model_type))


def create_model(num_dense_layers_base, num_dense_nodes_base,
                 num_dense_layers_end, num_dense_nodes_end,
                 activation, adam_b1, adam_b2, adam_eps):

    # linear stack of layers
    model = keras.Sequential()

    # input layer
    model.add(keras.layers.Dense(
        num_input_nodes,  # number of nodes
        input_shape=(num_input_nodes,)  # tuple specifying data input dimensions only needed in first layer
             ))

    # n number of hidden layers (base, i.e. first layers):
    for i in range(num_dense_layers_base):
        model.add(keras.layers.Dense(
            num_dense_nodes_base,
            activation=activation
        ))

    # n number of hidden layers (end, i.e. last layers):
    for i in range(num_dense_layers_end):
        model.add(keras.layers.Dense(
            num_dense_nodes_end,
            activation=activation
        ))

    # output layer
    model.add(keras.layers.Dense(1, activation=keras.activations.linear))

    # Adam optimiser
    optimizer = tf.keras.optimizers.Adam(
        lr=0.0001,  # learning rate
        beta_1=adam_b1,  # exponential decay rate for the first moment estimates
        beta_2=adam_b2,  # exponential decay rate for the second-moment estimates (
        epsilon=adam_eps  # prevent any division by zero
    )

    # compile model
    model.compile(
        loss='mae',  # loss function
        optimizer=optimizer,  # optimisaion function defined above
        metrics=["mae"]  # metric to be recorded
    )

    return model


def train_model(train_set):
    """
    1. Unpack training data.
    2. Define hyper-perameter ranges.
    3. Define early stopping perameters.
    4. Optimise hyper-perameters and save best model.
    5. Save mae per call to CSV.
    """

    # seperate features and labels and convert to numpy array
    X = train_set.drop(offset_col_name, axis=1).to_numpy()
    y = train_set.pop(offset_col_name).to_numpy()

    # retrieve data sets
    train_X, validate_X, train_y, validate_y = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # init recording statistics
    with open(output_dr + 'DNN_statistics.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['MAE (kcal/mol)', 'parameters'])

    # define hyper-perameters
    # layers
    dim_num_dense_layers_base = Integer(low=1, high=2, name='num_dense_layers_base')
    dim_num_dense_nodes_base = Categorical(categories=list(np.linspace(5, 261, 10, dtype=int)),
                                           name='num_dense_nodes_base')
    dim_num_dense_layers_end = Integer(low=1, high=2, name='num_dense_layers_end')
    dim_num_dense_nodes_end = Categorical(categories=list(np.linspace(5, 261, 10, dtype=int)),
                                          name='num_dense_nodes_end')

    # optimiser
    dim_adam_b1 = Categorical(categories=list(np.linspace(0.8, 0.99, 11)), name='adam_b1')
    dim_adam_b2 = Categorical(categories=list(np.linspace(0.8, 0.99, 11)), name='adam_b2')
    dim_adam_eps = Categorical(categories=list(np.linspace(0.0001, 0.5, 11)), name='adam_eps')

    dimensions = [dim_num_dense_layers_base, dim_num_dense_nodes_base,
                  dim_num_dense_layers_end, dim_num_dense_nodes_end,
                  dim_adam_b1, dim_adam_b2, dim_adam_eps]

    # Set early stopping variable to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',  # monitor validation loss
        mode='min',  # monitoring loss
        patience=20,  # large patience for small batch size
        verbose=0)  # do not output to terminal

    # start hyper-perameter optimisation
    @use_named_args(dimensions=dimensions)
    def fitness(num_dense_layers_base, num_dense_nodes_base,
                num_dense_layers_end, num_dense_nodes_end,
                adam_b1, adam_b2, adam_eps):

        # Create the neural network with these hyper-parameters:
        model = create_model(num_dense_layers_base=num_dense_layers_base,
                             num_dense_nodes_base=num_dense_nodes_base,
                             num_dense_layers_end=num_dense_layers_end,
                             num_dense_nodes_end=num_dense_nodes_end,
                             activation=tf.keras.activations.relu,
                             adam_b1=adam_b1, adam_b2=adam_b2, adam_eps=adam_eps)

        history = model.fit(train_X, train_y, # training data
                            epochs=epochs,  # number of forward and backward runs
                            validation_data=(validate_X, validate_y),  # validation data
                            verbose=1,  # input progress to terminal
                            callbacks=[early_stopping],  # prevent over fitting
                            batch_size=30)  # increase efficiency

        # update statistics
        mae = history.history['val_mae'][-1]
        parameters = [num_dense_layers_base, num_dense_nodes_base,
                      num_dense_layers_end, num_dense_nodes_end,
                      adam_b1, adam_b2, adam_eps]
        with open(output_dr + 'DNN_statistics.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([mae, parameters])

        print('\nMAE = {} kcal/mol\n'.format(mae))
        print('Parameters: {}\n'.format(parameters))

        # If the regressor accuracy of the saved model is improved ...
        global best_mae
        if mae < best_mae:
            # save the new model to harddisk.
            model.save(output_dr + 'ddGhydr_' + model_type + '_model.h5')
            # Update the regressor accuracy.
            best_mae = mae

        # Delete the Keras model with these hyper-parameters from memory.
        del model

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()

        return mae

    # a place for optimiser to start looking
    default_parameters = [2, 261, 1, 61, 0.857, 0.933, 0.20006]

    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=n_calls,
                                x0=default_parameters)

    # save skopt object and analyse in a separate script as
    # https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/bayesian-optimization.ipynb
    # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
    dump(search_result, output_dr + 'gp_minimize_result.pickle', store_objective=False)
    logging.info('Saved {}gp_minimize_result.pickle.'.format(output_dr))

    logging.info('Final parameters: {}.'.format(search_result.x))
    return search_result


# https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script started on {}'.format(time.ctime()))

    # prevent accidental clearance of previous models in output/
    question = 'Are you sure you want to clear saved models in ./output/ ??'
    query = query_yes_no(question, 'no')
    if not query:
        print('Script terminated.')
        sys.exit()
    elif query:
        print('./output/ cleared.')

    # clean slate output_dr
    if os.path.exists(output_dr):
        shutil.rmtree(output_dr)
    if not os.path.exists(output_dr):
        os.mkdir(output_dr)

    main()

    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Script finished.')
