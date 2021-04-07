import numpy as np
import keras, random, time
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from fl_serverAT import obj_to_pickle_string, pickle_string_to_obj
import datasource
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from Paras import GPU_MEMORY_FRACTION_P, POINT_TRANSFER_P

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION_P
set_session(tf.Session(config=config))


class LocalModel(object):
    def __init__(self, model_config, data_collected):
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])
        train_data, test_data, valid_data = data_collected
        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data])
        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data])
        self.x_valid = np.array([tup[0] for tup in valid_data])
        self.y_valid = np.array([tup[1] for tup in valid_data])
        self.point_transfer = POINT_TRANSFER_P

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def set_weights_transfer(self, new_weights):
        tmp_weights = self.get_weights()
        tmp_weights[:self.point_transfer] = new_weights[:self.point_transfer]
        self.model.set_weights(tmp_weights)

    def train_one_round(self):
        self.model.fit(self.x_train, self.y_train,
                       epochs=self.model_config['epoch_per_round'],
                       batch_size=self.model_config['batch_size'],
                       verbose=0,
                       validation_split=0.0, validation_data=None)
        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return self.model.get_weights(), score[0], score[1]

    def validate(self):
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        return score

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score


class FederatedClient(object):
    def __init__(self, value_data_class_train_size, server_host, server_port, datasource):
        self.my_class_distr = value_data_class_train_size['class_distr']
        self.train_size = value_data_class_train_size['train_size']
        self.local_model = None
        self.datasource = datasource()
        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.register_handles()
        print('sent wakeup')
        self.sio.emit('client_wake_up')
        self.sio.wait()

    def on_init(self, *args):
        model_config = args[0]
        fake_data, my_class_distr = self.datasource.fake_non_iid_data_with_class_train_size(self.my_class_distr,
                                                                                            self.train_size,
                                                                                            data_split=model_config[
                                                                                                'data_split'])
        self.local_model = LocalModel(model_config, fake_data)
        self.sio.emit('client_ready', {
            'train_size': self.local_model.x_train.shape[0],
            'class_distr': my_class_distr
        })

    def register_handles(self):
        #########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            print('update requested')
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            if req['full_communication_validation']:
                self.local_model.set_weights(weights)
            else:
                self.local_model.set_weights_transfer(weights)
            my_weights, train_loss, train_accuracy = self.local_model.train_one_round()
            resp = {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'full_communication': req['full_communication_validation'],
            }
            if req['run_validation']:
                valid_loss, valid_accuracy = self.local_model.validate()
                resp['valid_loss'] = valid_loss
                resp['valid_accuracy'] = valid_accuracy
            self.sio.emit('client_update', resp)

        def on_test_and_eval(*args):
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            buf_weights = self.local_model.get_weights()
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.local_model.set_weights(buf_weights)
            self.sio.emit('client_eval', resp)

        def on_stop_and_eval(*args):
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.sio.emit('client_eval', resp)
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('test_and_eval', on_test_and_eval)
        self.sio.on('stop_and_eval', on_stop_and_eval)

    def intermittently_sleep(self, p=.1, low=10, high=36):
        if random.random() < p:
            time.sleep(random.randint(low, high))


if __name__ == "__main__":
    FederatedClient("127.0.0.1", 5000, datasource.Mnist)
