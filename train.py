#! /usr/bin/python
# -*- coding: utf8 -*-
import os

import numpy as np
import tensorflow as tf

from deepsleep.trainer import DeepFeatureNetTrainer, DeepSleepNetTrainer
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 20,
                           """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 100,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_integer('finetune_epochs', 200,
                            """Number of epochs for fine-tuning DeepSleepNet.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")

tf.app.flags.DEFINE_boolean('loop_all', True,
                            """Whether to train all subjects iteratively.""")



def pretrain(n_epochs, fold_idx):
    trainer = DeepFeatureNetTrainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds, 
        fold_idx=fold_idx,
        batch_size=100, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return pretrained_model_path


def finetune(model_path, n_epochs, fold_idx):
    trainer = DeepSleepNetTrainer(
        data_dir=FLAGS.data_dir, 
        output_dir=FLAGS.output_dir, 
        n_folds=FLAGS.n_folds, 
        fold_idx=fold_idx, 
        batch_size=10, 
        input_dims=EPOCH_SEC_LEN*100, 
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path, 
        n_epochs=n_epochs, 
        resume=FLAGS.resume
    )
    return finetuned_model_path


def main(argv=None):

    # if loop_all is True, n_folds used for loop range and fold_idx
    # used for loop end index.
    # if loop_all is False, fold_idx used to specify the training index

    if FLAGS.loop_all is True:
        n_folds = FLAGS.n_folds
    else:
        n_folds = 1

    for i in range(n_folds):
        # if not loop, fall back to fold_idx
        if FLAGS.loop_all is False:
            i = FLAGS.fold_idx

        # Output dir
        output_dir = os.path.join(FLAGS.output_dir, "fold{}".format(i))
        if not FLAGS.resume:
            if tf.gfile.Exists(output_dir):
                tf.gfile.DeleteRecursively(output_dir)
            tf.gfile.MakeDirs(output_dir)

        pretrained_model_path = pretrain(
            n_epochs=FLAGS.pretrain_epochs,
            fold_idx=i
        )
        finetuned_model_path = finetune(
            model_path=pretrained_model_path,
            n_epochs=FLAGS.finetune_epochs,
            fold_idx=i
        )
        print("==== fold {} done ====".format(i))


if __name__ == "__main__":
    tf.app.run()
