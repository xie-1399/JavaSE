# coding: utf-8

# In[1]:


import os
import numpy as np

import tensorflow as tf


# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------

train_dir = '/media/xxl/98D5D1E9544F330D/Python/Package_Recongise/train'
beibao = []
label_beibao = []
bianzhidai = []
label_bianzhidai = []
laganxiang = []
label_laganxiang = []
shoutibao = []
label_shoutibao = []
yingerche = []
label_yingerche = []
zhixiangzi = []
label_zhixiangzi = []
chongwuxiang = []
label_chongwuxiang = []


# In[2]:


def get_files(file_dir):
    for file in os.listdir(file_dir + '/beibao'):
        beibao.append(file_dir + '/beibao' + '/' + file)
        label_beibao.append(0)
    for file in os.listdir(file_dir + '/bianzhidai'):
        bianzhidai.append(file_dir + '/bianzhidai' + '/' + file)
        label_bianzhidai.append(1)
    for file in os.listdir(file_dir + '/laganxiang'):
        laganxiang.append(file_dir + '/laganxiang' + '/' + file)
        label_laganxiang.append(2)
    for file in os.listdir(file_dir + '/shoutibao'):
        shoutibao.append(file_dir + '/shoutibao' + '/' + file)
        label_shoutibao.append(3)
    for file in os.listdir(file_dir + '/yingerche'):
        yingerche.append(file_dir + '/yingerche' + '/' + file)
        label_yingerche.append(4)
    for file in os.listdir(file_dir + '/zhixiangzi'):
        zhixiangzi.append(file_dir + '/zhixiangzi' + '/' + file)
        label_zhixiangzi.append(5)
    for file in os.listdir(file_dir + '/chongwuxiang'):
        chongwuxiang.append(file_dir + '/chongwuxiang' + '/' + file)
        label_chongwuxiang.append(6)

    image_list = np.hstack((beibao, bianzhidai, laganxiang, shoutibao, yingerche, zhixiangzi, chongwuxiang))
    label_list = np.hstack((label_beibao, label_bianzhidai, label_laganxiang, label_shoutibao, label_yingerche,
                            label_zhixiangzi, label_chongwuxiang))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list, label_list

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    n_sample = len(all_label_list)
    n_val = 100  # 测试样本数
    n_train = 600  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    print(val_images)
    return tra_images, tra_labels, val_images, val_labels


# In[3]:
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch