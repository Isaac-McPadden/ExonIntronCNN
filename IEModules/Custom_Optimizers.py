import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class AccumOptimizer(tf.keras.optimizers.Adam):
    """Adam with gradient accumulation."""

    def __init__(self, accum_steps=4, **kwargs):
        super().__init__(**kwargs)
        self.accum_steps = accum_steps
        self._gradients_accum = []      # buffer for accumulating grads
        self._current_vars = None       # to hold vars until we apply

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"accum_steps": self.accum_steps})
        return cfg

    @tf.function
    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        # unpack
        grads, vars = zip(*grads_and_vars)

        # initialize buffer on first call
        if not self._gradients_accum:
            self._gradients_accum = [tf.zeros_like(g) for g in grads]

        # accumulate
        self._gradients_accum = [
            acc_g + g for acc_g, g in zip(self._gradients_accum, grads)
        ]

        # count one “micro” step
        self.iterations.assign_add(1)

        # store the vars so _apply_accumulated can see them
        self._current_vars = vars

        # every accum_steps, apply and reset; otherwise do nothing
        return tf.cond(
            tf.equal(self.iterations % self.accum_steps, 0),
            true_fn=self._apply_accumulated,
            false_fn=tf.no_op,
        )

    @tf.function
    def _apply_accumulated(self):
        # average the accumulated gradients
        avg_grads = [
            g / tf.cast(self.accum_steps, g.dtype)
            for g in self._gradients_accum
        ]

        # perform the real Adam update
        super(AccumOptimizer, self).apply_gradients(
            zip(avg_grads, self._current_vars)
        )

        # reset the buffer
        self._gradients_accum = [
            tf.zeros_like(g) for g in self._gradients_accum
        ]

        return None