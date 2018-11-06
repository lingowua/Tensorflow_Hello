import tensorflow as tf

weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

with tf.Session() as sess:

    # 输出为（|1|+|-2|+|-3|+|4|）*0.5 = 5 。其中0.5为正则化的权重。
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))

    # 输出为（1方+|-2|方+|-3|方+|4|方）/2 *0.5 = 7.5 。其中0.5为正则化的权重。
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
