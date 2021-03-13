import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()
tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
observations = f(observation_index_points) + np.random.normal(0., .05, 50)

amplitude = tfp.util.TransformedVariable(
    1., tfb.Exp(), dtype=tf.float64, name='amplitude')
length_scale = tfp.util.TransformedVariable(
    1., tfb.Exp(), dtype=tf.float64, name='length_scale')
kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)
observation_noise_variance = tfp.util.TransformedVariable(
  np.exp(-5), tfb.Exp(), name='observation_noise_variance')

gp = tfd.GaussianProcess(
  kernel=kernel,
  index_points=observation_index_points,
  observation_noise_variance=observation_noise_variance)

optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)


@tf.function
def optimize():
    with tf.GradientTape() as tape:
        loss = -gp.log_prob(observations)
    grads = tape.gradient(loss, gp.trainable_variables)
    optimizer.apply_gradients(zip(grads, gp.trainable_variables))
    return loss


for i in range(1000):
    neg_log_likelihood_ = optimize()
    if i % 100 == 0:
          print("Step {}: NLL = {}".format(i, neg_log_likelihood_))
