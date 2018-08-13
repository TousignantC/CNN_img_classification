from prepare_data import *
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
#np.set_printoptions(threshold=np.nan)

trainDSG = DataSetGenerator("./train")
testDSG = DataSetGenerator("./test")
photosDSG = DataSetGenerator("./2016test")


height = 48
width = 48
classes = 5

# Import training data
train_data = trainDSG.get_mini_batches(batch_size=150, image_size=(height, width), allchannel=True)
for data in train_data:
    X = data[0]/255
    Y = data[1]

# Import testing data
test_data = testDSG.get_mini_batches(batch_size=25, image_size=(height, width), allchannel=True)
for data in test_data:
    Xtest = data[0]/255
    Ytest = data[1]


# Import data to predict
image_list, image_name = [], []
photo_path = "./2016test/*.jpg"
for filename in glob.glob(photo_path):
    if os.path.getsize(filename) > 0:
        image_name.append(filename[len(photo_path)-5:])
        im = cv2.imread(filename)
        im = resizeAndPad(img=im, size=(height, width))
        im = im.reshape(1, height, width, 3)/255
        image_list.append(im)
Xphotos = np.vstack(image_list)
print(len(Xphotos))
Yphotos = np.zeros((len(Xphotos), classes))
print(Xphotos.shape)
print(Yphotos.shape)


# Creating model
x = tf.placeholder(tf.float32, shape=[None, height, width, 3])
y_true = tf.placeholder(tf.float32, shape=[None, classes])
hold_prob = tf.placeholder(tf.float32)

# Helper Functions


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# Create layers
convo_1 = convolutional_layer(x, shape=[4, 4, 3, height])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, height, height*2])
convo_2_pooling = max_pool_2by2(convo_2)

k = int(height/4)**2 * height*2
convo_2_flat = tf.reshape(convo_2_pooling, [-1, k])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, classes)

# Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

#init = tf.global_variables_initializer()

# Graph session

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 200
    for i in range(epochs):
        #batch = ch.next_batch(100)
        sess.run(train, feed_dict={x: X, y_true: Y, hold_prob: 0.5})

        # PRINT OUT A MESSAGE EVERY 10 STEPS
        if i % 10 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: Xtest, y_true: Ytest, hold_prob: 1.0}))
            print('\n')



    saver.save(sess, 'models/model.ckpt')
    # saver.restore(sess, 'models/model48.ckpt')

    # Use the model to get predicted y
    #result = sess.run(y_pred, feed_dict={x: Xphotos, y_true: Yphotos, hold_prob: 1.0})
    result = sess.run(y_pred, feed_dict={x: Xphotos, hold_prob: 1.0})
    result_list = list(result)


text_file = open("output/Output.txt", "w")
for i in range(len(image_name)):
    text_file.write("{} {},{}\n".format(i+1, image_name[i], result_list[i].argmax()))

text_file.close()




