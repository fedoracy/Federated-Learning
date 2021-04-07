from fl_clien_T import FederatedClient
import datasource
from multiprocessing import Pool
from Paras import MIN_NUM_WORKERS_P, VALUE_DATA_CLASS_TRAIN_SIZE_P


def start_client(value_data_class_train_size):
    print("start client")
    c = FederatedClient(value_data_class_train_size, "127.0.0.1", 5000, datasource.Mnist)


if __name__ == '__main__':
    p = Pool(MIN_NUM_WORKERS_P)
    p.map(start_client, VALUE_DATA_CLASS_TRAIN_SIZE_P)