import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class AccumOptimizer(tf.keras.optimizers.Adam):
    """Adam with gradient accumulation, using tf.Variables in build()."""

    def __init__(self, accum_steps: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.accum_steps   = accum_steps
        self.accum_vars    = None     # will hold tf.Variable buffers
        self._current_vars = None     # store vars between micro-steps

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"accum_steps": self.accum_steps})
        return cfg

    def build(self, var_list):
        # Called once automatically when first needed
        super().build(var_list)
        # Create one non-trainable buffer per model variable
        self.accum_vars = [
            tf.Variable(
                initial_value=tf.zeros_like(v),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.MEAN,
                name=f"accum_{v.name.replace(':', '_')}"
            )
            for v in var_list
        ]

    @tf.function
    def _apply_accumulated(self):
        # Average the accumulated grads
        avg_grads = [
            acc / tf.cast(self.accum_steps, acc.dtype)
            for acc in self.accum_vars
        ]
        # Apply them just once
        super().apply_gradients(zip(avg_grads, self._current_vars))
        # Reset buffers to zero
        for acc in self.accum_vars:
            acc.assign(tf.zeros_like(acc))
        return tf.constant(True)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads, vars_ = zip(*grads_and_vars)

        # Lazily build buffers
        if self.accum_vars is None:
            self.build(vars_)

        # 1️⃣ Accumulate into our buffers
        for g, buf in zip(grads, self.accum_vars):
            if g is not None:
                buf.assign_add(g)

        # 2️⃣ Store the vars for the deferred apply
        self._current_vars = vars_

        # 3️⃣ Every accum_steps, do the real Adam update
        apply_now = tf.equal((self.iterations + 1) % self.accum_steps, 0)
        tf.cond(apply_now,
                self._apply_accumulated,      # true branch
                lambda: tf.constant(False))   # no-op branch

        # 4️⃣ Always advance the “micro-batch” iteration counter
        self.iterations.assign_add(1)