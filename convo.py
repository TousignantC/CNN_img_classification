import glob
from utils import *

#np.set_printoptions(threshold=np.nan)

trainDSG = DataSetGenerator("./train")
testDSG = DataSetGenerator("./test")

learning_rate = 0.001
epochs = 200

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

print("Importing data to predict...")
# Import data to predict
image_list, image_name = [], []
photo_path = "./photos/*.jpg"
for filename in glob.glob(photo_path):
    if os.path.getsize(filename) > 0:
        image_name.append(filename[len(photo_path)-5:])
        im = cv2.imread(filename)
        im = resizeAndPad(img=im, size=(height, width))
        im = im.reshape(1, height, width, 3)/255
        image_list.append(im)
Xphotos = np.vstack(image_list)
#Yphotos = np.zeros((len(Xphotos), classes))


print("Creating model...")
# Creating model
x = tf.placeholder(tf.float32, shape=[None, height, width, 3])
y_true = tf.placeholder(tf.float32, shape=[None, classes])
hold_prob = tf.placeholder(tf.float32)


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
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cross_entropy)

print("Graph session...")
# Graph session
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # for i in range(epochs):
    #     sess.run(train, feed_dict={x: X, y_true: Y, hold_prob: 0.5})
    #     if i % 10 == 0:
    #         print('Currently on step {}'.format(i))
    #         print('Accuracy is:')
    #         # Test the Train Model
    #         matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    #
    #         acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    #
    #         print(sess.run(acc, feed_dict={x: Xtest, y_true: Ytest, hold_prob: 1.0}))
    #         print('\n')
    #
    # saver.save(sess, 'models/model.ckpt')
    saver.restore(sess, 'models/model.ckpt')

    # Use the model to get predicted y
    logits = sess.run(y_pred, feed_dict={x: Xphotos, hold_prob: 1.0})
    preds = tf.argmax(logits, axis=1)
    results = preds.eval()

print("Creating .txt file with results...")
text_file = open("output/Output.txt", "w")
for i in range(len(image_name)):
    text_file.write("{},{}\n".format(image_name[i], results[i]))
text_file.close()
