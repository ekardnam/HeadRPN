import tensorflow as tf

def rpn_accuracy(p_true, p_pred):
    """
        Accuracy metric for the RPN classifier. It's basically a binary accuracy
        but it removes the -1 tokens from the ground truth values which signify
        reagions are being deactivated.
        Args:
            p_true, the ground truth classes in batches
            p_pred, the predicted classes in batches
        Returns:
            the avarage binary accuracy between ground truth and predictions
            for each batch
    """
    valid_indices = tf.where(tf.not_equal(p_true, -1.0))
    p_true_valid = tf.gather_nd(p_true, valid_indices)
    p_pred_valid = tf.gather_nd(p_pred, valid_indices)
    return tf.keras.metrics.binary_accuracy(p_true_valid, p_pred_valid)
