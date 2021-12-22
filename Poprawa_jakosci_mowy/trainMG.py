#!/usr/bin/env python
import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
import librosa
import pickle
import yaml

from datetime import datetime

from datasetMG import Dataset
from networks.unet_dilated_tf2 import Network
from evaluation.reco import testsdr


def get_default_pars():
    pars = {'epochs' : 200,
            'batch_size' : 10,
            'learning_rate_decay' : 0.985,
            'speech_weight' : 0.75,
            'properties_file': './properties.yaml'}
    return pars

def fold(dataset, pars, out_path):
    num_epochs = pars['epochs']
    learning_rate_decay = pars['learning_rate_decay']
    speech_weight = pars['speech_weight']


    model = Network()
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate())

    @tf.function
    def train_fun(x, t, w):
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean((y - t)**2.0, axis=2, keepdims=True)*w
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    print('Total number of network\'s parameters: {}'.format(len(model.trainable_variables)))
    
    @tf.function
    def val_fun(x, t, w):
        y = model(x, training=False)
        loss = tf.reduce_mean((y - t)**2.0, axis=2, keepdims=True)*w
        loss = tf.reduce_mean(loss)
        return loss, y

    exp_name = os.getcwd().split('/')[-1]
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = '../logs_tb/'
    train_log_dir = log_path + current_time + '_' + exp_name
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    print("Starting training...")
    sys.stdout.flush()
    best_val_sdr = 0.0
    prev_train_err = 1000000.0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        backup = model.get_weights()
        for batch in dataset.batch_creator(mode='TRAIN'):
            i, t, w, _ = batch
            train_err += train_fun(i, t, w)
            train_batches += 1
        if train_err / train_batches > 1.1 * prev_train_err:
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("   training loss:  \t\t{:.6f}".format(train_err / train_batches))
            print("   catastophic event - restoring previous weights")
            model.set_weights(backup)
            old_lr = optimizer.lr.read_value()
            new_lr = old_lr * learning_rate_decay
            optimizer.lr.assign(new_lr)
            #learning_rate.set_value(lasagne.utils.floatX(learning_rate.get_value() * learning_rate_decay))
            continue
        else:
            prev_train_err = train_err / train_batches
        
        old_lr = optimizer.lr.read_value()
        new_lr = old_lr * learning_rate_decay
        optimizer.lr.assign(new_lr)
 


        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        val_sdr = 0.0
        for batch in dataset.batch_creator(mode='VAL'):
            i, t, w, d = batch
            err, pred = val_fun(i, t, w)
            val_err += err

            sdr, pred = calc_sdr(d, pred, i)
            val_sdr += sdr 
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("   training loss:  \t\t{:.6f}".format(train_err / train_batches))
        print("   val      loss:  \t\t{:.6f}".format(val_err / val_batches))
        print("   val       sdr:  \t\t{:.6f}".format(val_sdr / val_batches))

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_err/train_batches, step=epoch+1)
            tf.summary.scalar('val_loss', val_err/val_batches, step=epoch+1)
            tf.summary.scalar('val_sdr', val_sdr/val_batches, step=epoch+1)

        sys.stdout.flush()
        if val_sdr / val_batches > best_val_sdr:
            print("   new best val sdr!")
            best_val_sdr = val_sdr / val_batches

            epoch_postfix = str((epoch // 50)*50)
            model_data = model.get_weights() #lasagne.layers.get_all_param_values(net)
            np.savez(os.path.join(out_path, 'model_weights_' + epoch_postfix + '.npz'), model_data)
           
            test_sdr = 0.0
            test_batches = 0
            predictions = list()
            for batch in dataset.batch_creator(mode='TEST'):
                i, t, w, d = batch
                _, pred = val_fun(i, t, w)

                sdr, pred = calc_sdr(d, pred, i)
                test_sdr += sdr
                predictions.append(pred)
                test_batches += 1


            print("   test      sdr:  \t\t{:.6f}".format(test_sdr / test_batches))
            with train_summary_writer.as_default():
                tf.summary.scalar('test_sdr', test_sdr/test_batches, step=epoch+1)
            
            pickle.dump(predictions, open(os.path.join(out_path, 'predictions_' + epoch_postfix + '.pkl'), 'wb'))
            sys.stdout.flush()
    return predictions

def calc_sdr(original_data, predictions, inputs):
    x_mag, x_phase, y_mag, y_phase, start_idx = original_data

    predictions = predictions[0, :, :, start_idx:start_idx+x_mag.shape[1]]
    predictions = np.clip(predictions, -10.0 + 1e-6, 10.0 - 1e-6)
    m_real_pred = predictions[0]
    m_imag_pred = predictions[1]

    inputs = inputs[0, :, :, start_idx:start_idx+x_mag.shape[1]]
    in_real = inputs[0]
    in_imag = inputs[1]

    K = 10.0
    C = 0.1

    m_real_pred = -(1.0/C)*np.log((K-m_real_pred)/(K+m_real_pred))
    m_imag_pred = -(1.0/C)*np.log((K-m_imag_pred)/(K+m_imag_pred))

    x_pred = (m_real_pred*in_real - m_imag_pred*in_imag) + 1j*(m_real_pred*in_imag + m_imag_pred*in_real)
    x_pred *= 40.0

    x_mag_pred = y_mag.copy()
    x_mag_pred[:256] = np.abs(x_pred)

    x_phase_pred = y_phase.copy()
    x_phase_pred[:256] = np.angle(x_pred)

    return testsdr(x_mag_pred, x_phase_pred, x_mag, x_phase), x_pred

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(123)

    if len(sys.argv) == 1:
        pars = get_default_pars()
    elif len(sys.argv) == 2:
        pars = yaml.load(open(sys.argv[1], 'r'))

    paths = yaml.load(open(pars['properties_file'], 'r'))

    datetime_tag = datetime.now().isoformat()
    out_dir = os.path.join(paths['output_path'], datetime_tag)
    os.makedirs(out_dir)

    open('./exp.tag', 'wt').write(datetime_tag)

    TRAIN_HDF5_FILE = os.path.join(paths['data_path'], 'mixes063_tr.hdf5')
    TEST_HDF5_FILE = os.path.join(paths['data_path'], 'mixes063_tst.hdf5')
    OUTPUT_FILE = os.path.join(out_dir, "x_pred.pkl")

    batch_size = pars['batch_size']


    data_files = {'train_hdf5_file' : os.path.join(paths['data_path'], 'corpora_train_bt.hdf5'),
                  'test_hdf5_file' : os.path.join(paths['data_path'], 'corpora_test_bt.hdf5'),
                  'train_noise_file' : os.path.join(paths['data_path'], 'noise_tr.wav'),
                  'test_noise_file' : os.path.join(paths['data_path'], 'noise_tst.wav'),
                  'train_starts_file' : os.path.join(paths['data_path'], 'start_tr_bt.txt'),
                  'test_starts_file' : os.path.join(paths['data_path'], 'start_tst_bt.txt')}

    path = os.getcwd()
    path_2 = os.path.abspath(os.path.join(path, os.pardir))
    print(path_2)
    clean_train_path = path_2 + '/magister/Clean/corpora_train_bt.hdf5'
    clean_test_path  = path_2 + '/magister/Clean/corpora_test_bt.hdf5'
    noise_path = path_2 + '/magister/Noise'

    data_files_2 = {'train_hdf5_file': clean_train_path,
                    'test_hdf5_file': clean_test_path,
                    'noise': noise_path,
                                                }
    

    dataset = Dataset(data_files_2, batch_size, Network)
    predictions = fold(dataset, pars, out_dir)
    pickle.dump(predictions, open(OUTPUT_FILE, "wb"))
    print('koniec')
