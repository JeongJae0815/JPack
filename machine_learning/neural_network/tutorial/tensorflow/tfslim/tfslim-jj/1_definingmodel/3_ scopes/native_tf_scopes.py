# -*- coding: utf-8 -*-
__modifier__='jeongjae0815@gmail.com'
import tensorflow as tf
import matplotlib.pyplot as plt

"""
name_scope()와 variable_scope()의 차이 비교
"""


def scoping(fn, scope1, scope2, vals):
    with fn(scope1):
        a = tf.Variable(vals[0], name='a')
        b = tf.get_variable('b', initializer=vals[1])
        c = tf.constant(vals[2], name='c')
        with fn(scope2):
            d = tf.add(a * b, c, name='res')

        print '\n  '.join([scope1, a.name, b.name, c.name, d.name]), '\n'
    return d


d1 = scoping(tf.variable_scope, 'scope_vars', 'res', [1, 2, 3])
d2 = scoping(tf.name_scope, 'scope_name', 'res', [1, 2, 3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./tmp/tf-slim-tutorial')
    writer.add_graph(sess.graph)
    print sess.run([d1, d2])
    writer.close()
