import tensorflow as tf
import numpy as np
# tf.compat.v1.disable_eager_execution()
def main():
    tf_2()
def tf_1():
    coefficient = np.array([[1.0], [-10], [25]])
    w = tf.Variable(0, dtype=tf.float32)
    x = tf.compat.v1.placeholder(tf.float32, [3, 1])
    cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]
    train = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.compat.v1.global_variables_initializer()
    session = tf.compat.v1.Session()
    session.run(init)
    
    for i in range(1000):
        session.run(train, feed_dict={x : coefficient})
    print(session.run(w))

def tf_2():
    w = tf.Variable(1, dtype=tf.float32, name="w")
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            J = tf.add(tf.add(w**2, tf.multiply(w, -10)), 25)
        grads = tape.gradient(J, [w])
        optimizer.apply_gradients(zip(grads, [w]))
    tf.print(w)

main()

