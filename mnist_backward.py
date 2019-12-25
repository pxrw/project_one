import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
BATCH_SIZE = 200
#学习率衰减的原始值
LEARNING_RATE_BASE = 0.1
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化参数
REGULARIZER = 0.0001
# 训练轮数
STEPS = 50000
#这个使用滑动平均的衰减率
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
 
def backward(mnist):
   #一共有多少个特征,784行,一列
   x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
   y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
   # 给前向传播传入参数x和正则化参数计算出y的值
   y = mnist_forward.forward(x,REGULARIZER)
   # 初始化global—step,它会随着训练轮数增加
   global_step = tf.Variable(0,trainable=False)
 
   # softmax和交叉商一起运算的函数，logits传入是x*w,也就是y
   ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
   cem = tf.reduce_mean(ce)
   loss = cem + tf.add_n(tf.get_collection("losses"))
 
   learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples/BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase = True)
   #梯度下降
   train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
 
    # 滑动平均处理,可以提高泛华能力
   ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
   ema_op = ema.apply(tf.trainable_variables())
   # 将train_step和滑动平均计算ema_op放在同一个节点
   with tf.control_dependencies([train_step,ema_op]):
      train_op = tf.no_op(name="train")
        
   saver = tf.train.Saver()
 
   with tf.Session() as sess:
        
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
 
      for i in range(STEPS):
         # mnist.train.next_batch()函数包含一个参数BATCH_SIZE,表示随机从训练集中抽取BATCH_SIZE个样本输入到神经网络
         # next_batch函数返回的是image的像素和标签label
         xs,ys = mnist.train.next_batch(BATCH_SIZE)
         _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
         if i % 1000 == 0:
            print("Ater {} training step(s),loss on training batch is {} ".format(step,loss_value))
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
 
def main():
    
   mnist = input_data.read_data_sets("./data",one_hot = True)
   backward(mnist)
 
if __name__ == "__main__":
   main()