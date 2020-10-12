# -*- coding: utf-8 -*-
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, SeparableConv1D
from keras.layers import LSTM, Bidirectional
import utils.tools as utils
import encode_schema as tool
import cnn_models as cnnmodel
import multiple_cnn_models as multiple_cnn
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot

# ================gpu动态占用====================#
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# A "Best-fit with coalescing" algorithm, simplified from
#   a version of dlmalloc.
config.gpu_options.allocator_type = "BFC"
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# =================gpu动态占用====================#

if __name__ == "__main__":
    path = "../result/DNA/"
    if not os.path.exists(path):
        os.makedirs(path)

    start = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    fw_perf = open(path + "/cnn_network_1layer_" + start, "w")
    fw_perf.write(
        "acc"
        + ","
        + "precision"
        + ","
        + "npv"
        + ","
        + "sensitivity"
        + ","
        + "specificity"
        + ","
        + "mcc"
        + ","
        + "ppv"
        + ","
        + "auc"
        + ","
        + "pr"
        + "\n"
    )

    scores = []
    k_fold = 5
    for index in range(5):
        dataset = []
        dataset = tool.read_seq("data/encode/DNA_encode" + str(index))
        # print(dataset.shape[0])
        label = np.loadtxt("data/encode/class")
        # print(label.shape())
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=512)
        for ((train, test), k) in zip(skf.split(dataset, label), range(k_fold)):

            top_words = 20
            # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
            #            truncate and pad input sequences
            max_review_length = 750
            encoded_docs = [one_hot(d, 20) for d in dataset[train]]
            encoded_docs2 = [one_hot(d, 20) for d in dataset[test]]
            X_train = sequence.pad_sequences(
                encoded_docs, maxlen=max_review_length, padding="pre", value=0.0
            )
            X_test = sequence.pad_sequences(
                encoded_docs2, maxlen=max_review_length, padding="pre", value=0.0
            )
            # create the model
            embedding_vecor_length = 128
            # 构建网络
            model = Sequential()
            model.add(
                Embedding(21, embedding_vecor_length, input_length=max_review_length)
            )

            model.add(
                Convolution1D(
                    128,
                    8,
                    strides=1,
                    padding="valid",
                    activation="relu",
                    kernel_initializer="random_uniform",
                    name="convolution_1d_layer1",
                )
            )
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=3))
            model.add(Dropout(0.3))
            model.add(
                Convolution1D(
                    128,
                    8,
                    strides=1,
                    padding="valid",
                    activation="relu",
                    kernel_initializer="random_uniform",
                    name="convolution_1d_layer2",
                )
            )
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=3))
            model.add(Dropout(0.3))

            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Flatten())
            model.add(Dense(256, activation="sigmoid"))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(2, activation="softmax"))

            # 指定回调函数
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=3, verbose=1
            )

            early_stopping = EarlyStopping(
                monitor="val_loss", min_delta=0, patience=5, verbose=1
            )

            print(model.summary())
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

            y_train = utils.to_categorical(label[train])
            model.fit(
                X_train,
                y_train,
                validation_split=0.1,
                batch_size=64,
                epochs=200,
                verbose=1,
                callbacks=[reduce_lr, early_stopping],
            )

            # prediction probability
            y_test = utils.to_categorical(label[test])
            predictions = model.predict(X_test)

            predictions_prob = predictions[:, 1]
            auc_ = roc_auc_score(label[test], predictions_prob)
            pr = average_precision_score(label[test], predictions_prob)

            y_class = utils.categorical_probas_to_classes(predictions)
            # true_y_C_C=utils.categorical_probas_to_classes(true_y_C)
            true_y = utils.categorical_probas_to_classes(y_test)
            (
                acc,
                precision,
                npv,
                sensitivity,
                specificity,
                mcc,
                f1,
            ) = utils.calculate_performace(len(y_class), y_class, true_y)
            print("======================")
            print("======================")
            print(
                "Iter " + str(index) + ", " + str(k + 1) + " of " + str(k_fold) + "cv:"
            )
            print(
                "\tacc=%0.4f,pre=%0.4f,npv=%0.4f,sn=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f"
                % (acc, precision, npv, sensitivity, specificity, mcc, f1)
            )
            print("\tauc=%0.4f,pr=%0.4f" % (auc_, pr))

            fw_perf.write(
                str(acc)
                + ","
                + str(precision)
                + ","
                + str(npv)
                + ","
                + str(sensitivity)
                + ","
                + str(specificity)
                + ","
                + str(mcc)
                + ","
                + str(f1)
                + ","
                + str(auc_)
                + ","
                + str(pr)
                + "\n"
            )

            scores.append(
                [acc, precision, npv, sensitivity, specificity, mcc, f1, auc_, pr]
            )

    scores = np.array(scores)
    print(len(scores))
    print(
        "acc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)
    )
    print(
        "precision=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100)
    )
    print(
        "npv=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100)
    )
    print(
        "sensitivity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100)
    )
    print(
        "specificity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100)
    )
    print(
        "mcc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100)
    )
    print(
        "f1=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100)
    )
    print(
        "roc_auc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100)
    )
    print(
        "roc_pr=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[8] * 100, np.std(scores, axis=0)[8] * 100)
    )

    end = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    print("start: %s" % start)
    print("end: %s" % end)

    fw_perf.write(
        "acc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)
        + "\n"
    )
    fw_perf.write(
        "precision=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100)
        + "\n"
    )
    fw_perf.write(
        "npv=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100)
        + "\n"
    )
    fw_perf.write(
        "sensitivity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100)
        + "\n"
    )
    fw_perf.write(
        "specificity=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100)
        + "\n"
    )
    fw_perf.write(
        "mcc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100)
        + "\n"
    )
    fw_perf.write(
        "f1=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100)
        + "\n"
    )
    fw_perf.write(
        "roc_auc=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100)
        + "\n"
    )
    fw_perf.write(
        "roc_pr=%.2f%% (+/- %.2f%%)"
        % (np.mean(scores, axis=0)[8] * 100, np.std(scores, axis=0)[8] * 100)
        + "\n"
    )
    fw_perf.write("start: %s" % start + "\n")
    fw_perf.write("end: %s" % end)

    fw_perf.close()
