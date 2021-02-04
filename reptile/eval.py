"""
Helpers for evaluating models.
"""

from .reptile import Reptile
from .variables import weight_decay

# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             dataset,
             num_shots=5,
             num_test_shots=2,
             eval_inner_batch_size=None,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             weight_decay_rate=1,
             reptile_fn=Reptile):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         pre_step_op=weight_decay(weight_decay_rate))
    # total_correct = 0
    # for _ in range(num_samples):
        # total_correct += reptile.evaluate(dataset, model.input_ph, model.label_ph,
        #                                   model.minimize_op, model.predictions,
        #                                   num_shots=num_shots,
        #                                   num_test_shots=num_test_shots,
        #                                   inner_batch_size=eval_inner_batch_size,
        #                                   inner_iters=eval_inner_iters, replacement=replacement)

    total_acc, cat_acc_dict = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                    model.minimize_op, model.predictions,
                                    num_shots=num_shots,
                                    num_test_shots=num_test_shots,
                                    inner_batch_size=eval_inner_batch_size,
                                    inner_iters=eval_inner_iters, replacement=replacement)
    # return total_correct / (num_samples * 2)

    return total_acc, cat_acc_dict

