import tensorflow as tf
from CNNet import *
from datetime import datetime
import numpy as np 
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片

n_classes = 10
# learning_rate = 0.01

start_learning_rate = 0.1
decay_rate = 0.9
decay_step = 100

epoch = 60

def getdata(filename, epoch, batch_size, cpapcity):
    def parser(record):
        features = tf.parse_single_example(record,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })  # return image and label
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [64, 64, 3])
        # 转换为float32类型，并做归一化处理
        img = tf.cast(img, tf.float32)# * (1. / 255)
        label = tf.cast(features['label'], tf.int64)
        return img,label
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(cpapcity)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    # data_info = pd.DataFrame({'label':}})
    iterator = dataset.make_initializable_iterator()
    return iterator

def datasettry():
    iterator = getdata('./sox_single_train.tfrecords', 1, 10, 2429)
    img_batch, label_batch = iterator.get_next()
    #label_batch = tf.one_hot(label_batch, N_CLASSES, 1, 0)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        '''
        while(True):
            try:
                print(sess.run(label_batch))
            except tf.errors.OutOfRangeError:
                print("trying end!")
                break
        '''
        im = sess.run(img_batch)
        print(im.shape)
        lab = sess.run(label_batch)
        lab = lab[0]
        print(lab)
        im = im[0]
        print(im.shape)
        print(im)
        im = np.array(im)
        print(im)
        '''
        cv2.namedWindow("Image")   
        cv2.imshow("Image", im)   
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        '''
        plt.title("opencv Read:")
        plt.imshow(im, cmap='gray')
        plt.show()
        
        

def train():
    train_iterator = getdata('./sox_single_train.tfrecords', 1, 50, 1000)
    val_iterator = getdata('./sox_single_val.tfrecords', 1, 50, 1000)

    #train_iterator = getdata('train.tfrecords', 1, 32, 7997)
    #val_iterator = getdata('val.tfrecords', 1, 32, 2000)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='label')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype = tf.bool)

    current_epoch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(start_learning_rate, current_epoch, decay_steps = decay_step, decay_rate = decay_rate)


    img_batch, label_batch = train_iterator.get_next()
    label_batch = tf.one_hot(label_batch, n_classes, 1, 0)
    output = inference(x, keep_prob, n_classes, is_training)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss, global_step=current_epoch)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(y, 1)), tf.float32))

    val_img, val_label = val_iterator.get_next()
    val_label = tf.one_hot(val_label, n_classes, 1, 0)
    val_output = inference(x, keep_prob, n_classes, is_training)
    val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=val_output, labels=y))
    val_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_output,1), tf.argmax(y, 1)), tf.float32))
    
    acc_line = []
    time_line = []
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth = True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        sess.run(init)
        i = 0
        for ep in range(epoch):
            sess.run(train_iterator.initializer)
            time_line.append(ep)
            while(True):
                try:
                    current_epoch = i
                    i = i+1
                    batch_x, batch_y = sess.run([img_batch, label_batch])
                
                    _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.8, is_training: True})
                
                    if i%10 == 0:
                        learnrate = sess.run(learning_rate)
                        print(learnrate)
                        train_arr = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0, is_training: False})
                        print("%s: Step [%d]  Loss : %f, training accuracy :  %g" % (datetime.now(), i, loss_val, train_arr))
                    # 只指定了训练结束后保存模型，可以修改为每迭代多少次后保存模型
                    '''
                    if (i + 1) == MAX_STEP:
                        saver.save(sess, './model/model.ckpt')
                    '''
                except tf.errors.OutOfRangeError:
                    print("trainning end")
                    sess.run(val_iterator.initializer)
                    print('validating start:')
                    valset_acc = []
                    while(True):
                        try:
                            val_x, val_y = sess.run([val_img, val_label])
                            val_ls, val_acc = sess.run([val_loss, val_accuracy], feed_dict={x:val_x, y:val_y, keep_prob:1.0, is_training: False})
                            valset_acc.append(val_acc)
                            print("%s: Step [%d]  Loss : %f, valing accuracy :  %g" % (datetime.now(), i, val_ls, val_acc))
                        except tf.errors.OutOfRangeError:
                            print(np.mean(valset_acc))
                            acc_line.append(np.mean(valset_acc))
                            print("validating end")
                            break
                    break
        builder = tf.saved_model.builder.SavedModelBuilder('./model/model2/')
        builder.add_meta_graph_and_variables(sess, ['sox_single_classifier'])
        builder.save()
        plt.figure()
        plt.plot(time_line, acc_line)
        plt.savefig("./acc2.jpg")
if __name__ == '__main__':
    train()
    # datasettry()