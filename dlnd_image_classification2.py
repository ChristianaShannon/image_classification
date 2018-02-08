# coding: utf-8

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile
from sklearn import preprocessing
import pickle
import problem_unittests as tests
import helper


cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Use Floyd's cifar-10 dataset if present
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()

tests.test_folder_path(cifar10_dataset_folder_path)

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import helper
import numpy as np

# Explore the dataset
batch_id = 3
sample_id = 2
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    return x

tests.test_normalize(normalize)



enc = preprocessing.LabelBinarizer()
# labels = [0,1,2,3,4,5,6,7,8,9]
enc.fit(range(10))

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    return enc.transform(x)


tests.test_one_hot_encode(one_hot_encode)

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))

import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, (None,) + image_shape, name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, (None, n_classes), name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, name='keep_prob')


tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)


# init Weights
mean = 0.0
STD = 1/np.sqrt(32*32)


def weight_value(shape):
    #shape = [ksize[0],ksize[1],input_depth,output_depth]
    init = tf.truncated_normal(shape, mean = mean, stddev = STD)
    return tf.Variable(init)

def conv2d(x, W, b, strides):
    x = tf.nn.conv2d(x, W, strides=[1, strides[0], strides[1], 1], padding='SAME') 
    x = tf.nn.bias_add(x, b)
    return x
    

def maxpool2d(x, size, kstrides):
    return tf.nn.relu(tf.nn.max_pool(
        x,
        ksize=[1, size[0], size[1], 1],
        strides=[1, kstrides[0], kstrides[1], 1],
        padding='SAME'))

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    x_shape = x_tensor.get_shape().as_list() # x_shape =(None,32, 32, 3)
    input_depth = x_shape[3]
    W_shape = [conv_ksize[0], conv_ksize[1], input_depth, conv_num_outputs]
    
    W_conv = weight_value(W_shape)
    b_conv = tf.Variable(tf.zeros(conv_num_outputs))
    conv1 = conv2d(x_tensor,W_conv,b_conv,conv_strides)
    conv1 = maxpool2d(conv1, pool_ksize, pool_strides)
    return conv1 

tests.test_con_pool(conv2d_maxpool)


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    
    #basic function:
    x_shape = x_tensor.get_shape().as_list()
    dimensions = x_shape[1]*x_shape[2]*x_shape[3]
    return tf.reshape(x_tensor, [-1,dimensions])
    #return tf.contrib.layers.flatten(x_tensor)

tests.test_flatten(flatten)


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # use tf.contrib.layers
    # return tf.contrib.layers.fully_connected(x_tensor, num_outputs, tf.nn.relu)
    
    num_inputs = tf.cast(x_tensor.get_shape().as_list()[1], tf.int32)
    W = weight_value([num_inputs,num_outputs])
    b = tf.Variable(tf.zeros(num_outputs))
    
    fully_y = tf.nn.relu(tf.add(tf.matmul(x_tensor,W), b))
    return fully_y

tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    #use tf.contrib.layers
    #return tf.contrib.layers.linear(x_tensor, num_outputs)
    num_inputs = tf.cast(x_tensor.get_shape().as_list()[1], tf.int32)
    W = weight_value([num_inputs,num_outputs])
    b = tf.Variable(tf.zeros(num_outputs))
    
    #output_y = tf.nn.relu(tf.add(tf.matmul(x_tensor,W), b))
    output_y = tf.add(tf.matmul(x_tensor,W), b)
    return output_y

tests.test_output(output)

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """

    conv_num_outputs = [32, 64, 128]
    conv_ksize = [(3,3), (3,3), (3,3)]
    conv_strides = [(1,1), (1,1), (1,1)]
    pool_ksize = [(2,2), (3,3), (2,2)]
    pool_strides = [(1,1), (2,2), (2,2)]
    full_outputs = 512
    last_outputs = 10
    
    x_tensor = conv2d_maxpool(x, conv_num_outputs[0], conv_ksize[0], conv_strides[0], pool_ksize[0], pool_strides[0])
    x_tensor = conv2d_maxpool(x_tensor, conv_num_outputs[1], conv_ksize[1], conv_strides[1], pool_ksize[1], pool_strides[1])
    x_tensor = conv2d_maxpool(x_tensor, conv_num_outputs[2], conv_ksize[2], conv_strides[2], pool_ksize[2], pool_strides[2])

    x_flat = flatten(x_tensor)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers

    x_fully = fully_conn(x_flat, full_outputs)
    x_fully = tf.nn.dropout(x_fully, keep_prob)
    
    y_output = output(x_fully, last_outputs)
    return y_output

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """

    feed_dict = {x: feature_batch, y: label_batch, keep_prob: keep_probability}
    session.run(optimizer, feed_dict = feed_dict)

tests.test_train_nn(train_neural_network)


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    current_cost = sess.run(
        cost,
        feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})
    print('Cost: {:<8.4} Valid Accuracy: {:<5.4}'.format(
        current_cost,
        valid_accuracy))


epochs = 28
batch_size = 256
keep_probability = 0.5


print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
