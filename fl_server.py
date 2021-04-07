import pickle, keras, uuid
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import random, codecs, json, time
from flask import *
from flask_socketio import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from Paras import GPU_MEMORY_FRACTION_P, MIN_NUM_WORKERS_P, MAX_NUM_ROUNDS_P, NUM_CLIENTS_CONTACTED_PER_ROUND_P, \
    ROUNDS_BETWEEN_VALIDATIONS_P
from Paras import DATA_SPLIT_P, EPOCH_PER_ROUND_P, BATCH_SIZE_P

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION_P
set_session(tf.Session(config=config))


class GlobalModel(object):
    """docstring for GlobalModel"""

    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        self.prev_train_loss = None
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.training_start_time = int(round(time.time()))

    def build_model(self):
        raise NotImplementedError()

    def update_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights

    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.valid_losses += [[cur_round, cur_time, aggr_loss]]
        self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def aggregate_test_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.test_losses += [[cur_round, cur_time, aggr_loss]]
        self.test_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies,
            "test_loss": self.test_losses,
            "test_accuracy": self.test_accuracies
        }


def aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes):
    total_size = np.sum(client_sizes)
    aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                       for i in range(len(client_sizes)))
    aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                             for i in range(len(client_sizes)))
    return aggr_loss, aggr_accuraries


class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


class FLServer(object):
    MIN_NUM_WORKERS = MIN_NUM_WORKERS_P
    MAX_NUM_ROUNDS = MAX_NUM_ROUNDS_P
    NUM_CLIENTS_CONTACTED_PER_ROUND = NUM_CLIENTS_CONTACTED_PER_ROUND_P
    ROUNDS_BETWEEN_VALIDATIONS = ROUNDS_BETWEEN_VALIDATIONS_P

    def __init__(self, global_model, host, port):
        self.global_model = global_model()
        self.ready_client_sids = set()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.model_id = str(uuid.uuid4())
        self.current_round = -1
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):
        @self.socketio.on('connect')
        def handle_connect():
            print("{}  connected:".format(request.sid))

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print("{}  reconnected:".format(request.sid))

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print("{}  disconnected:".format(request.sid))
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: {}".format(request.sid))
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,

                'data_split': DATA_SPLIT_P,
                'epoch_per_round': EPOCH_PER_ROUND_P,
                'batch_size': BATCH_SIZE_P
            })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training: {}, {}".format(request.sid, data))
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            print("received client update of bytes: {} \nhandle client_update: {}".format(sys.getsizeof(data),
                                                                                          request.sid))

            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                if len(self.current_round_client_updates) == FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )
                    print("aggr_train_loss: {} \naggr_train_accuracy: {}".format(aggr_train_loss, aggr_train_accuracy))
                    if 'valid_loss' in self.current_round_client_updates[0]:
                        aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                            [x['valid_loss'] for x in self.current_round_client_updates],
                            [x['valid_accuracy'] for x in self.current_round_client_updates],
                            [x['valid_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )
                        print("aggr_valid_loss: {} \naggr_valid_accuracy: {}".format(aggr_valid_loss,
                                                                                     aggr_valid_accuracy))
                        self.valid_and_eval()
                    self.global_model.prev_train_loss = aggr_train_loss
                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval: {} \neval_resp: {}".format(request.sid, data))
            self.eval_client_updates += [data]
            if len(self.eval_client_updates) == len(self.ready_client_sids):
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_test_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                    self.current_round
                );
                print("\naggr_test_loss: {} \naggr_test_accuracy: {} \n##########== overall_test ==##########".format(
                    aggr_test_loss, aggr_test_accuracy))
                self.eval_client_updates = []

    def train_next_round(self):
        self.current_round += 1
        self.current_round_client_updates = []
        print("###Round {} ###".format(self.current_round))
        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print("request updates from: {}".format(client_sids_selected))
        for rid in client_sids_selected:
            emit('request_update', {
                'model_id': self.model_id,
                'round_number': self.current_round,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle',
                'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
            }, room=rid)

    def valid_and_eval(self):
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('valid_and_eval', {
                'model_id': self.model_id,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def stop_and_eval(self):
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                'model_id': self.model_id,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


if __name__ == '__main__':
    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000");
    server.start()
