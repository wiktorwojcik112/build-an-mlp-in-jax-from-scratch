"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    return jax.random.split(key, num=num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    return jnp.argmax(inputs[:, :num_classes], axis=1)

# Step 6 - one_hot_encode_labels
import jax
import jax.numpy as jnp

def one_hot_encode_labels(labels, num_classes):
    def generate_label_arr(label, num_classes):
        arr = jnp.zeros((num_classes,))
        arr = arr.at[label].set(1.0)
        return arr

    return jax.vmap(lambda label: generate_label_arr(label, num_classes))(labels)

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    W = sample_normal_matrix(key, (in_dim, out_dim)) * scale
    b = jnp.zeros(shape=(out_dim,))

    return { 'W': W, 'b': b }

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    layer_len = len(layer_sizes)
    subkeys = split_prng_key(key, layer_len)
    layers = []

    for i, (key, size) in enumerate(zip(subkeys, layer_sizes[:layer_len - 1])):
        nxt_size = layer_sizes[i + 1]

        layers += [init_linear_layer(key, size, nxt_size, scale=scale)]

    return layers

# Step 9 - linear_forward (not yet solved)
# TODO: implement

# Step 10 - relu_activation (not yet solved)
# TODO: implement

# Step 11 - softmax_probabilities (not yet solved)
# TODO: implement

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

