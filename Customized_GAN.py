from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os

"""
Parameters
activation1 : activation function in generator
- {"relu", "leaky_relu", "elu", "selu"}, default="relu"
activation2 : activation function in discriminator
- {"relu", "leaky_relu", "elu", "selu"}, default="relu"
num_hidden_layer1 : number of hidden layer in generator
- int, default=3
num_hidden_layer2 : number of hidden layer in discriminator
- int, default=3
solver : solfer for weight optimization
- {"rmsprop", "adam"}, default="adam"
batch_size : size of minibatches for optimizers
- int, default=100
learning_rate : learning rate
- float, default=0.001
num_epochs : maximum number of iterations
- int, default=2000
file_name : name to be used for save
- string, default=None
"""

class GAN:
    def __init__(self, activation1="relu", activation2="relu", num_hidden_layer1=3, num_hidden_layer2=3, noise_dim=100, solver="adam", batch_size=100, learning_rate=0.001, num_epochs=2000, file_name=None):
        os.makedirs("./model/gan", exist_ok=True)
        tf.reset_default_graph()

        self.activation1 = activation1
        self.activation2 = activation2
        self.num_hidden_layer1 = num_hidden_layer1
        self.num_hidden_layer2 = num_hidden_layer2
        self.noise_dim = noise_dim
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.file_name = file_name

    def fit(self, input_data):
        self.input_data = input_data
        self.input_dim = len(input_data.columns)
        self.hidden_dim = int((self.input_dim + self.noise_dim) * 2 / 3)
        self.build_model()
        self.train()

    def generator(self, x, reuse=False):
        with tf.variable_scope("GAN/Generator", reuse=reuse):
            if self.activation1 == "relu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.relu)
                if self.num_hidden_layer1 != 1:
                    for i in range(self.num_hidden_layer1 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.relu)
            elif self.activation1 == "leaky_relu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.leaky_relu)
                if self.num_hidden_layer1 != 1:
                    for i in range(self.num_hidden_layer1 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.leaky_relu)
            elif self.activation1 == "elu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.selu)
                if self.num_hidden_layer1 != 1:
                    for i in range(self.num_hidden_layer1 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.elu)
            elif self.activation1 == "selu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.selu)
                if self.num_hidden_layer1 != 1:
                    for i in range(self.num_hidden_layer1 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.selu)
            output = tf.layers.dense(hidden_layer, self.input_dim, activation=tf.nn.relu)
        return output

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("GAN/Discriminator", reuse=reuse):
            if self.activation2 == "relu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.relu)
                if self.num_hidden_layer2 != 1:
                    for i in range(self.num_hidden_layer2 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.relu)
            elif self.activation2 == "leaky_relu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.leaky_relu)
                if self.num_hidden_layer2 != 1:
                    for i in range(self.num_hidden_layer2 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.leaky_relu)
            elif self.activation2 == "elu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.selu)
                if self.num_hidden_layer2 != 1:
                    for i in range(self.num_hidden_layer2 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.elu)
            elif self.activation2 == "selu":
                hidden_layer = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.selu)
                if self.num_hidden_layer2 != 1:
                    for i in range(self.num_hidden_layer2 - 1):
                        hidden_layer = tf.layers.dense(hidden_layer, self.hidden_dim, activation=tf.nn.selu)
            output = tf.layers.dense(hidden_layer, 1, activation=tf.nn.sigmoid)
        return output

    def build_model(self):
        self.generator_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.discriminator_input = tf.placeholder(tf.float32, shape=[None, self.input_dim])

        self.generator_sample = self.generator(self.generator_input)

        discriminator_real = self.discriminator(self.discriminator_input)
        discriminator_fake = self.discriminator(self.generator_sample, reuse=True)

        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake, labels=tf.ones_like(discriminator_fake)))
        self.discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_real, labels=tf.ones_like(discriminator_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake, labels=tf.zeros_like(discriminator_fake)))

        if self.solver == "adam":
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.solver == "rmsprop":
            generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        generator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        discrminator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

        self.train_generator = generator_optimizer.minimize(self.generator_loss, var_list=generator_variables)
        self.train_discriminator = discriminator_optimizer.minimize(self.discriminator_loss, var_list=discrminator_variables)

    def train(self):
        loss_log_file = pd.DataFrame(columns=["Iteration", "Discriminator_Loss", "Generator_Loss"])
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':1}))
        tf.global_variables_initializer().run(session=sess)
        for i in range(1, self.num_epochs + 1):
            input_data = shuffle(self.input_data, random_state=0)
            batches = [input_data[k:k + self.batch_size] for k in range(0, input_data.shape[0], self.batch_size)]
            for idx, x_batch in enumerate(batches):
                if x_batch.shape[0] != self.batch_size:
                    z_batch = np.random.uniform(-1., 1., size=[x_batch.shape[0], self.noise_dim])
                else:
                    z_batch = np.random.uniform(-1., 1., size=[self.batch_size, self.noise_dim])
                feed_dict = {self.discriminator_input: x_batch, self.generator_input: z_batch}

                _, dl = sess.run([self.train_discriminator, self.discriminator_loss], feed_dict=feed_dict)
                _, gl = sess.run([self.train_generator, self.generator_loss],feed_dict=feed_dict)

                if (i % 10 == 0 or i == 1) and idx == 0:
                    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dl, gl))
                    loss_log_file = loss_log_file.append({"Iteration" : i, "Discriminator_Loss" : dl, "Generator_Loss" : gl}, ignore_index=True)
        saver = tf.train.Saver()
        if self.file_name == None:
            loss_log_file.to_csv("./model/gan/GAN_Loss_Logs.csv", index=False)
            saver.save(sess, "./model/gan/GAN_Model.ckpt")
        else:
            loss_log_file.to_csv("./model/gan/"+self.file_name+"_GAN_Loss_Logs.csv", index=False)
            saver.save(sess, "./model/gan/"+self.file_name+"_GAN_Model.ckpt")

    def generate(self, size=1000):
        os.makedirs("./result", exist_ok=True)

        z = np.random.uniform(-1., 1., size=[size, self.noise_dim])
        sess = tf.Session()
        saver = tf.train.Saver()
        if self.file_name == None:
            saver.restore(sess, "./model/gan/_GAN_Model.ckpt")
        else:
            saver.restore(sess, "./model/gan/"+self.file_name+"_GAN_Model.ckpt")
        generated_data = sess.run(self.generator_sample, feed_dict={self.generator_input: z})
        generated_data = pd.DataFrame(generated_data, columns=self.input_data.columns)
        if self.file_name == None:
            generated_data.to_csv("./result/GAN_generated_data_size_" + str(size) + ".csv", index=False)
        else:
            generated_data.to_csv("./result/"+self.file_name+"_GAN_generated_data_size_" + str(size) + ".csv", index=False)
        

def start():
    current_data = pd.read_csv("sample.csv", engine="python")
    current_file_name = "sample"

    GAN = GAN(file_name=current_file_name)
    GAN.fit(current_data)
    GAN.generate(8760)


