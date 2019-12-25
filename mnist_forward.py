import tensorflow as tf
 
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE = 500
 
# 定义神经网络参数,传入两个参数,一个是shape一个是正则化参数大小
def get_weight(shape,regularizer):
    # tf.truncated_normal截断的正态分布函数,超过标准差的重新生成
   w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
   if regularizer != None:
        # 将正则化结果存入losses中
      tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
   return w
 
# 定义偏置b,传入shape参数
def get_bias(shape):
    # 初始化为0
   b = tf.Variable(tf.zeros(shape))
   return b
 
# 定义前向传播过程,两个参数,一个是输入数据,一个是正则化参数
def forward(x,regularizer):
    # w1的维度就是[输入神经元大小,第一层隐含层神经元大小]
   w1 = get_weight([INPUT_NODE,LAYER_NODE],regularizer)
    # 偏置b参数,与w的后一个参数相同
   b1 = get_bias(LAYER_NODE)
    # 激活函数
   y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
 
   w2 = get_weight([LAYER_NODE,OUTPUT_NODE],regularizer)
   b2 = get_bias(OUTPUT_NODE)
   y = tf.matmul(y1,w2)+b2
   return y
 
   