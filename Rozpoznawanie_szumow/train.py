import tensorflow as tf
import datetime

from mydataset1 import Dataset
from network import MyNetwork
#from network import model_resnet

noise = 'Noise'
clean_speech_file = 'Clean/corpora_train_bt.hdf5'


@tf.function
def train_step(signals, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(signals, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)



@tf.function
def val_step(signals, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(signals, training=False)
    v_loss = loss_object(labels, predictions)

    val_loss(v_loss)
    val_accuracy(labels, predictions)




if __name__ == "__main__":

    #tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    obiekt = Dataset(clean_speech_file, noise)
    classes = int(len(obiekt.noise)/3)
    print("classes - {}".format(classes))

    model = MyNetwork(classes)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    EPOCHS = 100

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch in obiekt.batch_creator('TRAIN', 10):
            i, o, labels, w = batch
            train_step(i, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
        )

        val_loss.reset_states()
        val_accuracy.reset_states()

        for batch in obiekt.batch_creator('VAL', 25):
            iv, ov, labelsv, w = batch
            val_step(iv, labelsv)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        print(
            f'Epoch_val {epoch + 1}, '
            f'Loss_val: {val_loss.result()}, '
            f'Accuracy_val: {val_accuracy.result() * 100}, '
        )

    print('koniec')
