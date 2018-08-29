import os
import resnet
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

num_class = 10
batch_size = 32
image_size = 28
input_size = 224
num_channel = 1
learning_rate = 0.01
num_iter = 1000

inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channel], name="input")
labels = tf.placeholder(tf.int32, shape=[None, num_class], name="label")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

logits = resnet.resnet_18(tf.image.resize_images(inputs, (input_size, input_size)), num_class, is_training)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
predict = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))
train = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist')
data_sets = input_data.read_data_sets(test_path, one_hot=True)

for step in range(num_iter):
    train_x, train_y = data_sets.train.next_batch(batch_size)
    train_x = train_x.reshape((batch_size, 28, 28, 1))
    _, accuracy_result, loss_result = sess.run([train, accuracy, loss], feed_dict={inputs: train_x, labels: train_y,
                                                                                   is_training: True})
    print('%d train accuracy: %f, loss: %f' % (step, accuracy_result, loss_result))

    if step % 100 == 0 and step > 0:
        valid_x, valid_y = data_sets.validation.next_batch(batch_size)
        valid_x = valid_x.reshape((batch_size, 28, 28, 1))
        accuracy_result, loss_result = sess.run([accuracy, loss],
                                                feed_dict={inputs: valid_x, labels: valid_y,
                                                           is_training: True})
        print('%d validation accuracy: %f, loss: %f' % (step, accuracy_result, loss_result))
