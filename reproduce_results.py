#!/usr/bin/env python

import value_nets
import mlc_datasets

def run_bibtex():
    """Code to reproduce the results on the Bibtex dataset"""
    learning_rate=0.1
    weight_decay=0.001
    inf_lr=0.5

    train_labels, train_features, _ = mlc_datasets.get_bibtex('train')
    num_epochs=150
    net = value_nets.ValueNetwork('./bibtex_dvn',
                                  feature_dim=train_features.shape[1],
                                  label_dim=train_labels.shape[1],
                                  learning_rate=learning_rate,
                                  inf_lr=inf_lr,
                                  num_hidden=150,
                                  weight_decay=weight_decay,
                                  include_second_layer=False)

    net.train(train_features, train_labels, train_ratio=0.95, epochs=num_epochs)
    net.reduce_learning_rate()
    net.train(train_features, train_labels, train_ratio=0.95, epochs=num_epochs)
    net.reduce_learning_rate()
    net.train(train_features, train_labels, train_ratio=0.95, epochs=int(num_epochs/2.0))

    # Train more, with the full training set
    net.train(train_features, train_labels, train_ratio=1, epochs=int(num_epochs * .1))

    test_labels, test_features, _, __ = mlc_datasets.get_bibtex('test')

    # Evaluate the final model
    mlc_datasets.evaluate_f1(net.predict, test_features, test_labels)

    return net

def run_bookmarks():
    """Code to reproduce the results on the Bookmarks dataset"""
    weight_decay=0.0001
    learning_rate=0.1
    inf_lr=0.750000

    train_labels, train_features, _ = mlc_datasets.get_bookmarks('train')
    num_epochs=250
    net = value_nets.ValueNetwork('./bookmarks_dvn',
                                  feature_dim=train_features.shape[1],
                                  label_dim=train_labels.shape[1],
                                  learning_rate=learning_rate,
                                  inf_lr=inf_lr,
                                  num_hidden=150,
                                  weight_decay=weight_decay,
                                  include_second_layer=True)

    net.train(train_features, train_labels, train_ratio=0.95, epochs=num_epochs)
    net.reduce_learning_rate()
    net.train(train_features, train_labels, train_ratio=0.95, epochs=int(num_epochs * .2))

    # Train more, with the full training set
    net.train(train_features, train_labels, train_ratio=1, epochs=int(num_epochs * .1))

    test_labels, test_features, _ = mlc_datasets.get_bookmarks('test')

    # Evaluate the final model
    mlc_datasets.evaluate_f1(net.predict, test_features, test_labels)

    return net


if __name__=='__main__':
    run_bibtex()


