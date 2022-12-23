# Keras

## Models API

There are three ways to define Keras model. According to the [documentation](https://keras.io/api/models/).

- Model class: Model group's layers into an object with training and inference features. Versatile way of create graphs of layers (ex. last layer with first or to the middle), not neccesarily linear as the Sequential class.
  - Functional API:
    You start from Input, you chain layer calls to specify the model's forward pass, and finally you create your model from inputs and outputs. Example:

    ```python
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
      ```

  - Subclassing the Model class
    You should define your layers in __init__() and you should implement the model's forward pass in call(). Example:

    ```python
    import tensorflow as tf

    class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
        x = self.dropout(x, training=training)
        return self.dense2(x)

    model = MyModel()
    ```

- Sequential class: Sequential groups a linear stack of layers into a tf. keras.Model.
    Example:

    ```python
    # Optionally, the first layer can receive an `input_shape` argument:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
    # Afterwards, we do automatic shape inference:
    model.add(tf.keras.layers.Dense(4))

    # This is identical to the following:
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(8))

    # Note that you can also omit the `input_shape` argument.
    # In that case the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dense(4))
    # model.weights not created yet

    # Whereas if you specify the input shape, the model gets built
    # continuously as you are adding layers:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
    model.add(tf.keras.layers.Dense(4))
    len(model.weights)
    # Returns "4"

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dense(4))
    model.build((None, 16))
    len(model.weights)
    # Returns "4"

    # Note that when using the delayed-build pattern (no input shape specified),
    # the model gets built the first time you call `fit`, `eval`, or `predict`,
    # or the first time you call the model on some input data.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)
    ```


## Miscellaneous Functions

* ```tf.squeeze(dens)``` Remove all dimension of length 1
* ```tf.sign(var)``` return -1, 0, or 1 depending the var sign.
* ```tf.reduce_max(3D_tensor, reduction_indices=2)``` return a 2D tensor, where only the max element in the 3dim is kept.
* ```tf.unstack(value, axis=0)```: If given an array of shape (A, B, C, D), and an axis=2, it will return a list of |C| tensor of shape (A, B, D).
* ```tf.nn.moments(x, axes)```: return the mean and variance of the vector in the dimension=axis
* ```tf.nn.xw_plus_b(x, w, b)```: explicit
* tf.global_variables(): return every new variables that are shred across machines in a distributed environment. Each time a Variable() constructor is called, it adds a new variabl ot he graph collection
* tf.convert_to_tensor(args, dtype): (tf.convert_to_tensor([[1, 2],[2, 3]], dtype=tf.float32)): convert an numpy array, a python list or scalar, to a Tensor.
* ```tf.placeholder_with_default(defautl_output, shape)```: One can see a placeholder as an element in the graph that must be fed an output value with the feed dictionnary, however it is possible to define placeholder that take default value.
* ```tf.variable_scope(name_or_scope, default_name)```: if name_or_scope is None, then scope.name is default_name.
* ```tf.get_default_graph().get_operations()```: return all operations in the graph, operations can be filtered by scope then with the python function ```startwith```. It returns a list of tf.ops.Operation
* ```tf.expand_dims([1, 2], axis=1)``` return a tensor where the axis dimensions is expanded. Here the new shape will be (2) -> (2, 1)
* ``` tf.pad(image, [[16, 16], [16, 16], [0, 0]])```: pad a tensor. Here the tensor is a 3D tensor of shape (5, 4, 3) for example. Afterwards it will be of size (16 + 5 + 16, 16 + 4 + 16, 0 + 3 + 0), where zeros are add _upper_ and _after_ the current vector.
* ```tf.groups(op_1, op_2, op_3)``` can be pass to sess.run and it will run all operations (but it will not return any output, only computed operations) 
* ```tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)``` expects labels to be int32 of size (batchsize), where every element is an integer from 0 to nbclasses. logits should be a float32 vector of size (batchsize, nbclasses) with values in it are not probabilities (logit form, before softmax)
* ```tensor.get_shape().assert_is_compatible_with(shape=)```: Check if shape matched
* ```tf.random.categorical(logits, num_samples=1)```: Draws samples from a categorical distribution. `logits` is the log unnormalized distribution.
* ```tf.data.Dataset.from_tensor_slices(texts_as_ints)```: Create a Dataset from ndarray or list.
* ```tf.cond(pred, fn1, fn2)```: Given a condition, fn1 or fn2 (a callable) is return. Here is an example to return a rgb image if it isn't already one: 
    ```
    image = tf.cond(pred=tf.equal(tf.shape(image)[2], 3), fn2=lambda: tf.image.grayscale_to_rgb(image), fn1=lambda: image)
    ```
* FLAGS is an internal mecanism that allowed the same functionnality as argparse
* Create an operator to run in a sess that will clip values
  ```
  clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_value_min, clip_value_max)) for
                                         var in list_tf_variables]
  ``` .
