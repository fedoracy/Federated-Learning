from fl_client import FederatedClient
import datasource
from multiprocessing import Pool    # import multiprocessing
from Paras import MIN_NUM_WORKERS_P, VALUE_DATA_CLASS_TRAIN_SIZE_P, GPU_MEMORY_FRACTION_P
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(device_count={'GPU': 2})
config.gpu_options.visible_device_list = '0, 1'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION_P
set_session(tf.Session(config=config))


def start_client(value_data_class_train_size):
    device_id = value_data_class_train_size['device_id'] - 1
    print("start client on {}".format(device_id))
    with tf.device('/gpu:{}'.format(device_id)):
        c = FederatedClient(value_data_class_train_size, "127.0.0.1", 5000, datasource.Mnist)


if __name__ == '__main__':
    p_list1 = []
    p_list2 = []
    for i in VALUE_DATA_CLASS_TRAIN_SIZE_P:
        if i['device_id']==1:
            p_list1.append(i)
        else:
            p_list2.append(i)
    p = Pool(int(MIN_NUM_WORKERS_P/2))
    p.map(start_client, p_list1)
    p.join()
    p.close()
    p = Pool(int(MIN_NUM_WORKERS_P/2))
    p.map(start_client, p_list2)
    p.join()
    p.close()