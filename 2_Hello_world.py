import tensorflow as tf
hello = tf.constant("Hello World")

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")


g = tf.Graph()
with g.device('/gpu:0'):
    result = a + b
print(result)


sess = tf.Session()

print(a.graph is tf.get_default_graph())
print(sess.run(hello))
print(sess.run(result))
sess.close()
