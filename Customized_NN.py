import tensorflow as tf
import pandas as pd
import numpy as np
import os

class NN:
    def __init__(self, activation="relu", num_hidden_layer=5, solver="adam", batch_size=100, learning_rate=0.001, num_epochs=2000, file_name=None):
        os.makedirs("./model/nn", exist_ok=True)
        tf.reset_default_graph()

        self.activation = activation
        self.num_hidden_layer = num_hidden_layer
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.file_name = file_name

    def fit(self, data_x, data_y):
        self.train_x = data_x
        self.train_y = data_y
        self.input_dim = len(train_x.columns)
        self.hidden_dim = int((self.input_dim + 1) * 2 / 3)
        self.build_model()
        self.train()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        Layers = {}
        for i in range(1, self.num_hidden_layer + 2):
            if i == 1:
                shape_1 = self.input_dim
                shape_2 = self.hidden_dim
            elif i == self.num_hidden_layer + 1:
                shape_1 = self.hidden_dim
                shape_2 = 1
            else:
                shape_1 = self.hidden_dim
                shape_2 = self.hidden_dim
            Layer = {"weights": tf.get_variable("W" + str(i), shape=[shape_1, shape_2], initializer=tf.contrib.layers.xavier_initializer()),
                     "biases": tf.Variable(tf.random_normal([shape_2]))}
            Layers[i - 1] = Layer
        Out = {}
        Out[0] = {"layers": X}
        for i in range(0, self.num_hidden_layer + 1):
            weight = Layers[i]["weights"]
            bias = Layers[i]["biases"]
            if i == self.num_hidden_layer:
                layer = tf.matmul(Out[i]["layers"], weight) + bias
            else:
                if self.activation == "relu":
                    layer = tf.nn.relu(tf.matmul(Out[i]["layers"], weight) + bias)
                elif self.activation == "leaky_relu":
                    layer = tf.nn.leaky_relu(tf.matmul(Out[i]["layers"], weight) + bias)
                elif self.activation == "elu":
                    layer = tf.nn.elu(tf.matmul(Out[i]["layers"], weight) + bias)
                elif self.activation == "selu":
                    layer = tf.nn.selu(tf.matmul(Out[i]["layers"], weight) + bias)
            Out[i + 1] = {"layers": layer}

        cost = tf.losses.mean_squared_error(labels=self.Y, predictions=Out[self.num_hidden_layer + 1]["layers"])
        if self.solver == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        elif self.solver == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    def train(self):
        loss_log_file = pd.DataFrame(columns=["Iteration", "Loss"])
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':1}))
        sess.run(tf.global_variables_initializer())
        train_batch = int(len(train_dataset) / self.batch_size)

        for epoch in range(1, self.num_epochs+1):
            total_cost = 0
            for i in range(train_batch):
                batch_data = self.train_x.iloc[i * self.batch_size:(i + 1) * self.batch_size, :]
                label_data = self.train_y.iloc[i * self.batch_size:(i + 1) * self.batch_size]
                batch_data = batch_data.values
                label_data = np.reshape(label_data.values, (len(label_data.values), 1))
                feed_dict = {X: batch_data, Y: label_data}
                train_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                total_cost += train_cost * len(self.train_x.iloc[i * self.batch_size:(i + 1) * self.batch_size, :])
            batch_data = self.train_x.iloc[(i + 1) * self.batch_size:, :]
            label_data = self.train_y.iloc[(i + 1) * self.batch_size:]
            batch_data = batch_data.values
            label_data = np.reshape(label_data.values, (len(label_data.values), 1))
            feed_dict = {X: batch_data, Y: label_data}
            train_cost, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            total_cost += train_cost * len(self.train_x.iloc[(i + 1) * self.batch_size:, :])

            if (epoch % 10 == 0 or epoch == 1):
                print("Iterations: %d\t loss: %.4f" % (epoch, total_cost/len(self.train_x)))
                loss_log_file = loss_log_file.append({"Iteration" : i, "Loss" : total_cost/len(self.train_x)}, ignore_index=True)

        saver = tf.train.Saver()
        if self.file_name == None:
            loss_log_file.to_csv("./model/nn/NN_Loss_Logs.csv", index=False)
            saver.save(sess, "./model/nn/NN_Model.ckpt")
        else:
            loss_log_file.to_csv("./model/nn/"+self.file_name+"_NN_Loss_Logs.csv", index=False)
            saver.save(sess, "./model/nn/"+self.file_name+"_NN_Model.ckpt")

    def predict(self, data_x):
        os.makedirs("./result", exist_ok=True)
        self.test_x = data_x

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.file_name == None:
            saver.restore(sess, "./model/nn/NN_Model.ckpt")
        else:
            saver.restore(sess, "./model/nn/"+self.file_name+"_NN_Model.ckpt")

        Y_Pred = sess.run(Out[self.num_hidden_layer + 1]["layers"], feed_dict={X: self.test_x})
        Y_Pred = pd.DataFrame(Y_Pred, columns=["Prediction"])
        result.to_csv("./result/" + file_name + "_NN.csv", header=True, index=False)



def start():
    train_data = pd.read_csv("train.csv", engine="python")
    test_data = pd.read_csv("test.csv", engine="python")
    current_file_name = "sample"
    
    NN = NN(file_name=current_file_name)
    NN.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
    NN.predict(test_data)
