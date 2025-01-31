import numpy as np
import keras
import random
from keras.datasets import mnist
from keras import backend as K
from Paras import DATA_SPLIT_P, MAX_NUM_CLASSES_PER_CLIENT_P


class DataSource(object):
    def __init__(self):
        raise NotImplementedError()

    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()

    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()


class Mnist(DataSource):
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = MAX_NUM_CLASSES_PER_CLIENT_P

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x = np.concatenate([x_train, x_test]).astype('float32')
        self.y = np.concatenate([y_train, y_test])
        n = self.x.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]
        data_split = DATA_SPLIT_P
        num_train = int(n * data_split[0])
        num_test = int(n * data_split[1])
        self.x_train = self.x[0:num_train]
        self.x_test = self.x[num_train:num_train + num_test]
        self.x_valid = self.x[num_train + num_test:]
        self.y_train = self.y[0:num_train]
        self.y_test = self.y[num_train:num_train + num_test]
        self.y_valid = self.y[num_train + num_test:]
        self.classes = np.unique(self.y)

    def gen_dummy_non_iid_weights(self):
        self.classes = np.array(range(10))
        num_classes_this_client = random.randint(2, Mnist.MAX_NUM_CLASSES_PER_CLIENT + 1)
        classes_this_client = random.sample(self.classes.tolist(), num_classes_this_client)
        w = np.array([random.random() for _ in range(num_classes_this_client)])
        weights = np.array([0.] * self.classes.shape[0])
        for i in range(len(classes_this_client)):
            weights[classes_this_client[i]] = w[i]
        weights /= np.sum(weights)
        return weights.tolist()

    def post_process(self, xi, yi):
        if K.image_data_format() == 'channels_first':
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], 1)
        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0])
        return xi / 255., y_vec

    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        return {
            "train": [self.post_process(self.x[idx], self.y[idx]) for idx in idxs],
            "test": self.post_process(self.x[np.arange(n_train, n_train + n_test)],
                                      self.y[np.arange(n_train, n_train + n_test)])
        }

    def sample_single_non_iid(self, x, y, weight=None):
        chosen_class = np.random.choice(self.classes, p=weight)
        candidates_idx = np.array([i for i in range(y.shape[0]) if y[i] == chosen_class])
        idx = np.random.choice(candidates_idx)
        return self.post_process(x[idx], y[idx])

    def fake_non_iid_data_with_class_train_size(self, my_class_distr, train_size, data_split=(.6, .3, .1)):
        test_size = int(train_size / data_split[0] * data_split[1])
        valid_size = int(train_size / data_split[0] * data_split[2])
        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr) for _ in
                     range(train_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr) for _ in
                     range(valid_size)]
        print("done generating fake data")
        return ((train_set, test_set, valid_set), my_class_distr)

    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6, .3, .1)):
        my_class_distr = [1. / self.classes.shape[0] * self.classes.shape[0]] if Mnist.IID \
            else self.gen_dummy_non_iid_weights()
        train_size = random.randint(min_train, max_train)
        test_size = int(train_size / data_split[0] * data_split[1])
        valid_size = int(train_size / data_split[0] * data_split[2])
        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr) for _ in range(train_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr) for _ in range(valid_size)]
        print("done generating fake data")
        return ((train_set, test_set, valid_set), my_class_distr)


if __name__ == "__main__":
    m = Mnist()
    for _ in range(10):
        print(m.gen_dummy_non_iid_weights())
