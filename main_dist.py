import tensorflow as tf
from dist.worker import Worker

tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("agg", 0, "A Gradient")
FLAGS = tf.app.flags.FLAGS

worker = Worker(FLAGS)
tf.app.run(worker.work())

