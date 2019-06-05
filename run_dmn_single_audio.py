"""Evaluate E-VQA."""
import os
import argparse

import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import Series, DataFrame

from model.DMN_single_A import DMN_single_A
import config as cfg
import util.datasetWithAudio as dt
import util.metrics as metrics

import copy


def train(epoch, dataset, config, log_dir):
    """Train model for one epoch."""
    model_config = config['model']
    train_config = config['train']
    sess_config = config['session']

    with tf.Graph().as_default():
        model = DMN_single_A(model_config)
        model.build_inference()
        model.build_loss(train_config['reg_coeff'])
        model.build_train(train_config['learning_rate'])

        with tf.Session(config=sess_config) as sess:
            sum_dir = os.path.join(log_dir, 'summary')
            # create event file for graph
            if not os.path.exists(sum_dir):
                summary_writer = tf.summary.FileWriter(sum_dir, sess.graph)
                summary_writer.close()
            summary_writer = tf.summary.FileWriter(sum_dir)

            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if ckpt_path:
                print('load checkpoint {}.'.format(ckpt_path))
                saver.restore(sess, ckpt_path)
            else:
                print('no checkpoint.')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                sess.run(tf.global_variables_initializer())

            stats_dir = os.path.join(log_dir, 'stats')
            stats_path = os.path.join(stats_dir, 'train.json')
            if os.path.exists(stats_path):
                print('load stats file {}.'.format(stats_path))
                stats = pd.read_json(stats_path, 'records')
            else:
                print('no stats file.')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)
                stats = pd.DataFrame(columns=['epoch', 'loss', 'acc'])

            # train iterate over batch
            batch_idx = 0
            total_loss = 0
            total_acc = 0
            batch_total = np.sum(dataset.train_batch_total)

            while dataset.has_train_batch:
                vgg, c3d, vgg_conv5, vgg_conv5_3, mfcc, question, answer, question_len = dataset.get_train_batch()
                tmp_batch_size = len(question_len)
                input_len = np.zeros(tmp_batch_size)+20
                feed_dict = {
                    model.video_feature: mfcc,
                    model.question_encode: question,
                    model.answer_encode: answer,
                    model.question_len_placeholder: question_len,
                    model.video_len_placeholder: input_len,
                    model.keep_placeholder: train_config['keep_prob']
                }
                _, loss, acc, reg_loss, log_loss = sess.run(
                    [model.train, model.loss, model.acc, model.reg_loss, model.log_loss], feed_dict)
                total_loss += loss
                total_acc += acc
                if batch_idx % 100 == 0:
                    print('[TRAIN] epoch {}, batch {}/{}, loss {:.5f}, acc {:.5f}, reg_loss {:.5f}, log_loss {:.5f}.'.format(
                        epoch, batch_idx, batch_total, loss, acc, reg_loss, log_loss))
                batch_idx += 1

            loss = total_loss / batch_total
            acc = total_acc / batch_total
            print('\n[TRAIN] epoch {}, loss {:.5f}, acc {:.5f}.\n'.format(
                epoch, loss, acc))

            summary = tf.Summary()
            summary.value.add(tag='train/loss', simple_value=float(loss))
            summary.value.add(tag='train/acc', simple_value=float(acc))
            summary_writer.add_summary(summary, epoch)

            record = Series([epoch, loss, acc], ['epoch', 'loss', 'acc'])
            stats = stats.append(record, ignore_index=True)

            saver.save(sess, os.path.join(ckpt_dir, 'model.ckpt'), epoch)
            stats.to_json(stats_path, 'records')
            dataset.reset_train()
            return loss, acc


def val(epoch, dataset, config, log_dir):
    """Validate model."""
    model_config = config['model']
    sess_config = config['session']

    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]

    with tf.Graph().as_default():
        model = DMN_single_A(model_config)
        model.build_inference()

        with tf.Session(config=sess_config) as sess:
            sum_dir = os.path.join(log_dir, 'summary')
            summary_writer = tf.summary.FileWriter(sum_dir)

            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()

            stats_dir = os.path.join(log_dir, 'stats')
            stats_path = os.path.join(stats_dir, 'val.json')
            if os.path.exists(stats_path):
                print('load stats file {}.'.format(stats_path))
                stats = pd.read_json(stats_path, 'records')
            else:
                print('no stats file.')
                if not os.path.exists(stats_dir):
                    os.makedirs(stats_dir)
                stats = pd.DataFrame(columns=['epoch', 'acc'])

            # val iterate over examples
            correct = 0

            groundtruth_answer_list = [] 
            predict_answer_list = []
            while dataset.has_val_example:
                vgg, c3d, vgg_conv5, vgg_conv5_3, mfcc, question, answer, question_len = dataset.get_val_example()
                input_len = 20
                feed_dict = {
                    model.video_feature: [mfcc],
                    model.question_encode: [question],
                    model.question_len_placeholder: [question_len],
                    model.video_len_placeholder: [input_len],
                    model.keep_placeholder: 1.0
                }
                prediction = sess.run(model.prediction, feed_dict=feed_dict)
                prediction = prediction[0]
                if answerset[prediction] == answer:
                    correct += 1
                groundtruth_answer_list.append(answer)
                predict_answer_list.append(answerset[prediction])

            acc = correct * 1.0 / dataset.val_example_total
            WUPS_0_0 = metrics.compute_wups(groundtruth_answer_list,predict_answer_list,0.0)
            WUPS_0_9 = metrics.compute_wups(groundtruth_answer_list,predict_answer_list,0.9)
            WUPS_acc = metrics.compute_wups(groundtruth_answer_list,predict_answer_list,-1)
            print('[VAL] epoch {}, acc {:.5f}.\n'.format(epoch, acc))
            print('[VAL] epoch {}, WUPS@acc {:.5f}.\n'.format(epoch, WUPS_acc))
            print('[VAL] epoch {}, WUPS@0.0 {:.5f}.\n'.format(epoch, WUPS_0_0))
            print('[VAL] epoch {}, WUPS@0.9 {:.5f}.\n'.format(epoch, WUPS_0_9))

            summary = tf.Summary()
            summary.value.add(tag='val/acc', simple_value=float(acc))
            summary_writer.add_summary(summary, epoch)

            record = Series([epoch, acc], ['epoch', 'acc'])
            stats = stats.append(record, ignore_index=True)
            stats.to_json(stats_path, 'records')

            dataset.reset_val()
            return acc


def test(dataset, config, log_dir, question_type_dict):
    """Test model, output prediction as json file."""
    model_config = config['model']
    sess_config = config['session']

    question_type_correct_count = copy.deepcopy(question_type_dict)
    question_type_all_count = copy.deepcopy(question_type_dict)
    for k in question_type_dict:
        question_type_correct_count[k] = 0
        question_type_all_count[k] = 0

    answerset = pd.read_csv(
        os.path.join(config['preprocess_dir'], 'answer_set.txt'), header=None)[0]

    with tf.Graph().as_default():
        model = DMN_single_A(model_config)
        model.build_inference()

        with tf.Session(config=sess_config) as sess:
            ckpt_dir = os.path.join(log_dir, 'checkpoint')
            save_path = tf.train.latest_checkpoint(ckpt_dir)
            saver = tf.train.Saver()
            if save_path:
                print('load checkpoint {}.'.format(save_path))
                saver.restore(sess, save_path)
            else:
                print('no checkpoint.')
                exit()

            # test iterate over examples
            result = DataFrame(columns=['id', 'answer'])
            correct = 0

            groundtruth_answer_list = [] 
            predict_answer_list = []
            while dataset.has_test_example:
                vgg, c3d, vgg_conv5, vgg_conv5_3, mfcc, question, answer, example_id, question_len = dataset.get_test_example()
                input_len = 20
                feed_dict = {
                    model.video_feature: [mfcc],
                    model.question_encode: [question],
                    model.question_len_placeholder: [question_len],
                    model.video_len_placeholder: [input_len],
                    model.keep_placeholder: 1.0
                }
                prediction = sess.run(model.prediction, feed_dict=feed_dict)
                prediction = prediction[0]

                result = result.append({'id': example_id, 'answer': answerset[prediction]}, ignore_index=True)
                if answerset[prediction] == answer:
                    correct += 1
                    question_type_correct_count[question[0]] += 1
                question_type_all_count[question[0]] += 1

                groundtruth_answer_list.append(answer)
                predict_answer_list.append(answerset[prediction])

            result.to_json(os.path.join(log_dir, 'prediction.json'), 'records')
            acc = correct * 1.0 / dataset.test_example_total
            WUPS_0_0 = metrics.compute_wups(groundtruth_answer_list,predict_answer_list,0.0)
            WUPS_0_9 = metrics.compute_wups(groundtruth_answer_list,predict_answer_list,0.9)
            WUPS_acc = metrics.compute_wups(groundtruth_answer_list,predict_answer_list,-1)
            print('[TEST] acc {:.5f}.\n'.format(acc))
            print('[TEST], WUPS@acc {:.5f}.\n'.format(WUPS_acc))
            print('[TEST], WUPS@0.0 {:.5f}.\n'.format(WUPS_0_0))
            print('[TEST], WUPS@0.9 {:.5f}.\n'.format(WUPS_0_9))

            print('######## question type acc list ######### ')
            for k in question_type_dict:
                print(question_type_dict[k] + ' acc {:.5f}.'.format(question_type_correct_count[k] * 1.0 / question_type_all_count[k]))
                print('correct = {:d}, all = {:d}'.format(question_type_correct_count[k],question_type_all_count[k]))

            dataset.reset_test()
            return acc


def main():
    """Main script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,
                        help='train/test')
    parser.add_argument('--gpu', required=True,
                        help='gpu id')
    parser.add_argument('--log', required=True,
                        help='log directory')
    parser.add_argument('--dataset', required=True,
                        help='dataset name, msvd_qa/msrvtt_qa')
    parser.add_argument('--config', required=True,
                        help='config id')
    args = parser.parse_args()

    config = cfg.get('dmn_plus', args.dataset, args.config, args.gpu)


    if args.dataset == 'msvd_qa':
        dataset = dt.MSVDQA(config['train']['batch_size'], config['preprocess_dir'])
        question_type_dict = {0:'what', 2:'who', 23:'how', 96:'when', 226:'where'}
    elif args.dataset == 'msrvtt_qa':
        dataset = dt.MSRVTTQA(config['train']['batch_size'], config['preprocess_dir'])
        question_type_dict = {0:'what', 3:'who', 21:'how', 61:'when', 133:'where'}

    if args.mode == 'train':
        best_val_acc = -1
        val_acc = 0
        not_improved = -1

        for epoch in range(0, 30):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                not_improved = 0
            else:
                not_improved += 1
            if not_improved == 10:
                print('early stopping.')
                break

            train(epoch, dataset, config, args.log)
            val_acc = val(epoch, dataset, config, args.log)

    elif args.mode == 'test':
        print('start test.')
        test(dataset, config, args.log, question_type_dict)


if __name__ == '__main__':
    main()
