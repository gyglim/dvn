"""Deep Value Networks.

This code implementes the multi-label classification parts
of our paper "Learning Deep Value Networks to Evaluate and Refine Structured Outputs"
by Michael Gygli, Mohammad Norouzi, Anelia Angelova
"""
import sys
import os
import Queue
import numpy as np
import tensorflow as tf
import threading
import time

__author__ = "Michael Gygli, ETH Zurich"


class ValueNetwork(object):
    def __init__(self, data_dir, learning_rate=0.01, inf_lr=0.5,
                 feature_dim=1836, label_dim=159, binarize=False,
                 linear=False, weight_decay=0, regularizer=tf.nn.l2_loss, num_hidden=None,
                 nonlinearity=tf.nn.softplus, num_pairwise=16, include_second_layer=False):
        """Create a ValueNet object used for training and inference.

        Parameters
        ----------
        data_dir : basestring
            Where to store the model and logs for tensorboard
        learning_rate : float
            learning rate for updating the value network parameters
        inf_lr : float
            learning rate for the inference procedure
        feature_dim : int
            dimensionality of the input features
        label_dim : int
            dimensionality of the output labels
        binarize : bool
            Binarize the predicted values, before computing the f1 scores.
            This works better in some cases
        weight_decay : float
            the weight decay
        regularizer : tensorflow op
            type of the regularizer, e.g. l1 or l2
        nonlinearity : tensorflow op
            type of the non-linearity to use
        num_hidden : int
            number of hidden units for the linear part
        num_pairwise : int
            number of pairwise units for the global (label interactions) part
        linear : bool
            Only use the feed-forward part (linear in y)?
        include_second_layer : bool
            include a linear layer after the two-layer perceptron (as done in SPENs)
        """
        self.sess = tf.InteractiveSession()
        self.sentinel = object()
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.learning_rate = learning_rate
        self.current_step = 0
        self.inf_lr = inf_lr
        self.binarize = binarize

        if num_hidden:
            self.num_hidden = num_hidden
        else:
            # SPEN uses 150 see https://github.com/davidBelanger/SPEN/blob/master/mlc_cmd.sh (feature_hid_size)
            self.num_hidden = 200

        # Set number of neurons for label interactions
        # These capture (anti-)correlations in the labels
        # SPEN uses 16 see https://github.com/davidBelanger/SPEN/blob/master/mlc_cmd.sh (energy_hid_size)
        self.num_pairwise = num_pairwise

        # Include a linear layer at the end? Corresponds to B term in Eq (4) of SPEN paper
        self.include_second_layer = include_second_layer

        self.linear = linear
        self.weight_decay = weight_decay
        self.regularizer = regularizer
        self.nonlinearity = nonlinearity
        self.build_graph()

        # Create a summary writer
        if data_dir:
            self.data_dir = data_dir
            self.writer = tf.summary.FileWriter('%s/log/train' % self.data_dir)
            self.val_writer = tf.summary.FileWriter('%s/log/val' % self.data_dir)
            self.saver = tf.train.Saver(tf.global_variables(),
                                        max_to_keep=50)
        else:
            self.data_dir = None
            self.writer = None
            self.val_writer = None
            self.saver = None


        # Initialize variables
        tf.global_variables_initializer().run()

    def restore(self, path):
        """restore weights at `path`"""
        self.saver.restore(self.sess, path)
        self.mean = np.load("%s/mean.npy" % os.path.dirname(path))
        self.std = np.load("%s/std.npy" % os.path.dirname(path))

    def train(self, train_features, train_labels, epochs=1000, batch_size=20, train_ratio=0.9):
        """Train the value net.

        Parameters
        ----------
        train_features
            features used for training/validation
        train_labels
            the corresponding labels
        epochs : int
            the number of epochs to train for
        batch_size : int
            batch size to use
        train_ratio : float
            defines the ratio of the data used for training. The rest is used for validation

        Returns
        -------
        f1 score on validation at the end of training

        """
        f1_scores = None
        train_features = np.array(train_features, np.float32)
        self.mean = np.mean(train_features, axis=0).reshape((1, -1))
        self.std = np.std(train_features, axis=0).reshape((1, -1)) + 10 ** -6
        train_features -= self.mean
        train_features /= self.std

        np.save("%s/mean.npz" % self.data_dir, self.mean)
        np.save("%s/std.npz" % self.data_dir, self.std)

        # Split of some validation data
        if not hasattr(self, 'indices'):  # use existing splits if there are
            np.random.seed(0)
            self.indices = np.random.permutation(np.arange(len(train_features)))
        split_idx = int(len(train_features) * train_ratio)
        val_features = train_features[self.indices[split_idx:]]
        val_labels = train_labels[self.indices[split_idx:]]
        train_features = train_features[self.indices[:split_idx]]
        train_labels = train_labels[self.indices[:split_idx]]

        # Start training
        for epoch in range(0, epochs):
            start = time.time()
            print 'Starting epoch %d (it: %d)' % (epoch, self.current_step)
            sys.stdout.flush()

            # Randomize ordeer
            order = np.random.permutation(np.arange(len(train_features)))
            train_features = train_features[order]
            train_labels = train_labels[order]

            # Start threads to fill sample queue
            queue = self._generator_queue(train_features, train_labels, batch_size)
            while True:
                data = queue.get(timeout=10)
                if data is not self.sentinel:
                    # Do a training step to learn to corrently score the solution (predicted labels)
                    features, pred_labels, f1_scores = data
                    self.current_step, _, summary_str = self.sess.run(
                            [self.global_step, self.train_step, self.summary_op],
                            feed_dict={self.features_pl: features,
                                       self.labels_pl: pred_labels,
                                       self.gt_scores_pl: f1_scores})
                    if self.writer:
                        self.writer.add_summary(summary_str, self.current_step)
                        self.writer.flush()
                else:
                    break

            # store model at the end of each epoch
            if epoch % 10 == 0 and self.saver:
                self.saver.save(self.sess, '%s/weights' % self.data_dir, global_step=self.current_step)
            print "Epoch took %.2f seconds" % (time.time() - start)
            sys.stdout.flush()

            # Compute validation performance
            f1_scores = []

            for idx in range(0, len(val_features), batch_size):
                # Get a batch
                features = val_features[idx:min(len(val_features), idx + batch_size)]
                gt_labels = val_labels[idx:min(len(val_labels), idx + batch_size)]

                # Generate data (predicted labels and their true performance)
                # This is just for reporting to tensorboard
                _, f1 = self.generate_examples(features, gt_labels, train=False, val=True)
                f1_scores.extend(list(f1))

            if len(f1_scores) > 0:
                f1_scores = np.array(f1_scores)
                print 'Validation mean F1: %.3f' % np.mean(f1_scores[:, 1])
                sys.stdout.flush()
                summary_str = self.sess.run(self.gt_dist_op, feed_dict={self.gt_scores_pl: f1_scores})
                if self.val_writer:
                    self.val_writer.add_summary(summary_str, self.current_step)
                    self.val_writer.flush()

        if len(f1_scores) > 0:
            return np.mean(f1_scores[:, 1])
        else:
            return None

    def generate_examples(self, features, gt_labels, train=False, val=False):
        """Run inference to obtain examples."""
        init_labels = self.get_initialization(features)

        # In training: Generate adversarial examples 50% of the time
        if train and np.random.rand() >= 0.5:
            # 50%: Start from GT; rest: start from zeros
            gt_indices = np.random.rand(gt_labels.shape[0]) > 0.5
            init_labels[gt_indices] = gt_labels[gt_indices]
            pred_labels = self.inference(features, init_labels, gt_labels=gt_labels, num_iterations=1,
                                         learning_rate=self.inf_lr, )
            log = False

        # Otherwise: Run standard inference
        else:
            pred_labels = self.inference(features, init_labels,
                                         learning_rate=self.inf_lr)

            log = True

        # Score the predicted labels and return the labels and scores
        labels = np.zeros((gt_labels.shape[0], 2))
        labels[:, 1] = [self.gt_value(pred_labels[idx], gt_labels[idx], train=train) for idx in
                        np.arange(0, gt_labels.shape[0])]
        labels[:, 0] = 1 - labels[:, 1]

        if log and train:  # Log the true values of the inferred labels (this is the value networks true training performance)
            summary_str = self.sess.run(self.gt_dist_op, feed_dict={self.gt_scores_pl: labels})
            if train and self.writer:
                self.writer.add_summary(summary_str, self.current_step)

        return pred_labels, labels

    def gt_value(self, pred_labels, gt_labels, train=True):
        """Compute the ground truth value of some predicted labels."""

        if not train or self.binarize:
            pred_labels = np.array(pred_labels >= 0.5, np.float32)

        intersect = np.sum(np.min([pred_labels, gt_labels], axis=0))
        union = np.sum(np.max([pred_labels, gt_labels], axis=0))
        return 2 * intersect / float(intersect + max(10 ** -8, union))

    def get_initialization(self, features):
        """Get the initial output hypothesis"""
        if features.ndim == 1:
            features = features[None]

        return np.zeros([features.shape[0], self.label_dim])

    def predict(self, features, binarize=True, num_iterations=20):
        """
        Predict the labels for some features (single example).

        Parameters
        ----------
        features
        binarize : bool
            return the binarized results. If false: return scores
        num_iterations : int
            number of iterations at inference

        Returns
        -------
        labels
            the predicted labels or scores
        """
        init_labels = self.get_initialization(features)
        features = np.array(features, np.float64)
        features -= self.mean[0]
        features /= self.std[0]

        labels = self.inference(features.reshape(1, -1),
                                init_labels,
                                learning_rate=self.inf_lr,
                                num_iterations=num_iterations).flatten()
        if binarize:
            return labels >= 0.5
        else:
            return labels

    def inference(self, features, initial_labels, gt_labels=None, learning_rate=0, num_iterations=20):
        """Runs deep dream inference

        Parameters
        ----------
        features
        initial_labels
        gt_labels
        learning_rate
        num_iterations

        Returns
        -------
        inferred label vector
        """

        pred_labels = initial_labels
        if gt_labels is not None:
            gt_values = np.zeros((gt_labels.shape[0], 2))

        for idx in range(0, num_iterations):
            # Compute the gradient
            if gt_labels is not None:  # Adversarial
                gt_values[:, 1] = [self.gt_value(pred_labels[idx], gt_labels[idx]) for idx in
                                   np.arange(0, gt_labels.shape[0])]
                gt_values[:, 0] = 1 - gt_values[:, 1]

                if idx == 0 and self.writer:
                    gradient, summary_str = self.sess.run([self.gradient, self.inf_summary_op],
                                                          feed_dict={self.features_pl: features,
                                                                     self.labels_pl: pred_labels,
                                                                     self.gt_scores_pl: gt_values})
                else:
                    gradient = self.sess.run(self.gradient,
                                             feed_dict={self.features_pl: features,
                                                        self.labels_pl: pred_labels,
                                                        self.gt_scores_pl: gt_values})
            else:
                if idx == 0 and self.writer:
                    gradient, summary_str = self.sess.run([self.gradient, self.inf_summary_op],
                                                          feed_dict={self.features_pl: features,
                                                                     self.labels_pl: pred_labels})
                else:
                    gradient = self.sess.run(self.gradient,
                                             feed_dict={self.features_pl: features,
                                                        self.labels_pl: pred_labels})

            # Update the labels to improve the predicted value
            pred_labels += learning_rate * gradient

            # Project back to the valid range
            # pred_labels[np.isnan(pred_labels)] = 0
            pred_labels[pred_labels < 0] = 0
            pred_labels[pred_labels > 1] = 1

            if idx == 0 and self.writer:
                self.writer.add_summary(summary_str, self.current_step)

        summary_str = self.sess.run(self.label_summary_op,
                                    feed_dict={self.labels_pl: pred_labels})
        if self.writer:
            self.writer.add_summary(summary_str, self.current_step)

        # Return the inferred labels
        return pred_labels

    def reduce_learning_rate(self, factor=0.1):
        """Reduce the current learing rate by multipling it with `factor`"""
        self.learning_rate *= factor
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss),
                                                                                         global_step=self.global_step)

    def get_prediction(self, feature_input, label_input):
        """Builds a network to predict a scalar value."""

        W = tf.Variable(tf.random_normal([self.feature_dim, self.num_hidden],
                                         stddev=np.sqrt(2.0 / (self.feature_dim))))
        b = tf.Variable(tf.zeros([self.num_hidden]))

        # Add L_x regularization
        self.loss += self.weight_decay * self.regularizer(W)

        prediction1 = self.nonlinearity(tf.matmul(feature_input, W) + b)

        # Add another layer
        if self.include_second_layer:
            num_units = self.num_hidden
        else:
            num_units = self.label_dim
        W2 = tf.Variable(tf.random_normal([self.num_hidden, num_units],
                                          stddev=np.sqrt(2.0 / (self.num_hidden))))
        b2 = tf.Variable(tf.zeros([num_units]))

        prediction1 = tf.matmul(prediction1, W2) + b2
        self.loss += self.weight_decay * self.regularizer(W2)

        # Add what corresponds to the b term in SPEN
        # Eq. 4 in http://www.jmlr.org/proceedings/papers/v48/belanger16.pdf
        if self.include_second_layer:
            W3 = tf.Variable(tf.random_normal([num_units, self.label_dim],
                                              stddev=np.sqrt(2.0 / (num_units))))
            self.loss += self.weight_decay * self.regularizer(W3)
            prediction2 = tf.reduce_sum(self.labels_pl * tf.matmul(self.nonlinearity(prediction1), W3), axis=1)
        else:
            prediction2 = tf.reduce_sum(label_input * prediction1, axis=1)

        if not self.linear:  # Add global terms
            # We add global features
            # As done in "Structured Prediction Energy Networks", Eq (5)
            Wp = tf.Variable(tf.random_normal([self.label_dim, self.num_pairwise],
                                              stddev=np.sqrt(2.0 / (self.label_dim))))
            Wp2 = tf.Variable(tf.random_normal([self.num_pairwise, 1],
                                               stddev=np.sqrt(2.0 / (self.num_pairwise))))
            prior_prediction = tf.matmul(label_input, Wp)
            prior_prediction = self.nonlinearity(prior_prediction)
            prior_prediction = tf.squeeze(tf.matmul(prior_prediction, Wp2))

            prediction2 += prior_prediction
            self.loss += self.weight_decay * self.regularizer(Wp)
            self.loss += self.weight_decay * self.regularizer(Wp2)

        return prediction2

    def build_graph(self):
        """Build the model.

        It creates the network and all the operations needed for training and inference.
        Additionally it creates some ops for logging.

        Returns
        -------
        None
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.features_pl = tf.placeholder(tf.float32, [None, self.feature_dim])
        self.labels_pl = tf.placeholder(tf.float32, [None, self.label_dim])
        self.gt_scores_pl = tf.placeholder(tf.float32, [None, 2])
        self.loss = 0

        raw_prediction = self.get_prediction(self.features_pl, self.labels_pl)
        predicted_values = tf.sigmoid(raw_prediction)
        self.predicted_values = predicted_values
        self.loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=raw_prediction,
                                                             labels=self.gt_scores_pl[:, 1])

        # Gradient for generating adversarial examples
        self.adv_gradient = tf.gradients(self.loss, self.labels_pl)[0]

        # Gradient for inference (maximizing predicted value)
        self.gradient = tf.gradients(self.predicted_values, self.labels_pl)[0]
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss),
                                                                                         global_step=self.global_step)

        # Create summary operations
        summaries = [tf.summary.scalar('loss', tf.reduce_mean(self.loss)),
                     tf.summary.histogram('predicted_f1_scores', predicted_values)]
        self.gt_dist_op = tf.summary.scalar('gt_f1_scores', tf.reduce_mean(self.gt_scores_pl[:, 1]))
        self.summary_op = tf.summary.merge(summaries)
        self.inf_summary_op = tf.summary.histogram('gradient', self.gradient)
        self.label_summary_op = tf.summary.histogram('predicted_tag_probabilities', self.labels_pl)

    def _generator_queue(self, train_features, train_labels, batch_size, num_threads=5):
        queue = Queue.Queue(maxsize=20)

        # Build indices queue to ensure unique use of each batch
        indices_queue = Queue.Queue()
        for idx in np.arange(0, len(train_features), batch_size):
            indices_queue.put(idx)

        def generate():
            try:
                while True:
                    # Get a batch
                    idx = indices_queue.get_nowait()
                    features = train_features[idx:min(len(train_features), idx + batch_size)]
                    gt_labels = train_labels[idx:min(len(train_labels), idx + batch_size)]

                    # Generate data (predcited labels and their true performance)
                    pred_labels, f1_scores = self.generate_examples(features, gt_labels, train=True)
                    queue.put((features, pred_labels, f1_scores))
            except Queue.Empty:
                queue.put(self.sentinel)

        for _ in range(num_threads):
            thread = threading.Thread(target=generate)
            thread.start()

        return queue
