from tensorflow.python.platform import gfile
import tensorflow as tf

sess = tf.Session()

with gfile.FastGFile('model/first.pb','rb') as f:
    graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
sess.graph.as_default()
tf.import_graph_def(graph_def,name='')

sess.run(tf.global_variables_initializer())

print(sess.run('weight:0'))
print(sess.run('bias:0'))

input_x = sess.graph.get_tensor_by_name('X:0')

op = sess.graph.get_tensor_by_name('results:0')

ret = sess.run(op, feed_dict={input_x: 2})

print(ret)