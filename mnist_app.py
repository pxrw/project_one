import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_forward
import mnist_backward


# 定义加载使用模型进行预测的函数
def restore_model(testPicArr):
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)
        # 加载滑动平均模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("/home/tarena/PXR/own exercise/手写识别项目测试/model/")
            if ckpt and ckpt.model_checkpoint_path:
                # 恢复当前会话,将ckpt中的值赋值给w和b
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 执行图计算
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1



# 图片预处理函数
def pre_pic(picName):
    # 先打开传入的原始图片
    img = Image.open(picName)
    # 使用消除锯齿的方法resize图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 变成灰度图，转换成矩阵
    im_arr = np.array(reIm.convert("L"))
    threshold = 50  # 对图像进行二值化处理，设置合理的阈值，可以过滤掉噪声，让他只有纯白色的点和纯黑色点
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    # 将图像矩阵拉成1行784列，并将值变成浮点型（像素要求的仕0-1的浮点型输入）
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application(path):
    testPicArr = pre_pic(path)
    # 将处理后的结果输入到预测函数最后返回预测结果
    preValue = restore_model(testPicArr)
    return preValue
