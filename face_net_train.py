# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import importlib
import os
import time
import facenet
import argparse
import sys
from tensorflow.python.ops import data_flow_ops
from six.moves import xrange
import utils
import itertools
import models



# 实际上我们这里用inception_resnet结构，其实所谓facenet就只是提出了triplet的方法
def main(args):
    network = importlib.import_module(args.model_def)
    subdir = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    np.random.seed(seed=args.seed)

    train_set = utils.___get_dataset(args.file_exp)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    # 如果选择了预训练的模型的话，这里我们会选取facenet的网络作预训练
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')
        # input queue 其实就是 路径/label,三元组，也就是make好pair的
        # label的必要性?
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 4
        images_and_labels = []
        # 这里其实就是对数据进行预处理，通过从queue中获取到数据，读取路径下的图片并且预处理后做成一个list
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
            # print('image input successfully!')

        # 其实就是对于list里面进行获取batch
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        # 返回一个节点，那么就可以做到更新节点，每次产生新的batch
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        # 网络的参数设置，返回的是输出的dropout和fc（bottleneck）之后的输出，那么label到底作用在哪里
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        # embedding 其实就是对于softmax进行距离的l2正则化
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # 把我们输入进去的图片集得到的结果拆分为三元组
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)
        # get到了，把learning rate 坐成一个placeholder就可以动态调整learning rate啦
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        # 正则化损失是什么
        # 图构建过程当中的所有的回归损失？哪里加进去了呢
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})
        # print('variable inti successfully!')
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # 线程管理器
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        # print('begin to train!')
        with sess.as_default():
            # 读取预训练模型
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop

            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                # print('train another epoch')
                train(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                      batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                      input_queue, global_step,
                      embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                      args.embedding_size, anchor, positive, negative, triplet_loss)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
    return model_dir


def train(args, sess, datasets, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0
    print(len(datasets))
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        # print('trying to get lr')
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
        # print('lr loaded successfully!')
    step = 0
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        # print('get in')
        for dataset in datasets:
            # print('trying to sample')
            image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

            print('Running forward pass on sampled images: ', end='')
            start_time = time.time()
            nrof_examples = args.people_per_batch * args.images_per_person
            # labels其实就是没有意义的，指的是当前的类的label
            labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
            # 输入的图片其实一个不定长的一个list
            image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
            # embarray 其实就是一个矩阵
            emb_array = np.zeros((nrof_examples, embedding_size))
            # 这一epoch总共的batch数目
            nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
            for i in range(nrof_batches):
                # 这一batch的大小
                batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
                emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                           learning_rate_placeholder: lr,
                                                                           phase_train_placeholder: True})
                # print(lab)
                emb_array[lab, :] = emb
            print('%.3f' % (time.time() - start_time))

            # Select triplets based on the embeddings
            print('Selecting suitable triplets for training')
            triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                        image_paths, args.people_per_batch, args.alpha)
            selection_time = time.time() - start_time
            print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
                  (nrof_random_negs, nrof_triplets, selection_time))

            # Perform training on the selected triplets
            nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
            triplet_paths = list(itertools.chain(*triplets))
            labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
            triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
            sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
            nrof_examples = len(triplet_paths)
            train_time = 0
            i = 0
            emb_array = np.zeros((nrof_examples, embedding_size))
            loss_array = np.zeros((nrof_triplets,))
            summary = tf.Summary()
            step = 0
            while i < nrof_batches:
                start_time = time.time()
                batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
                feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
                             phase_train_placeholder: True}
                err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch],
                                                  feed_dict=feed_dict)
                # print(lab)
                emb_array[lab, :] = emb
                loss_array[i] = err
                duration = time.time() - start_time
                print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                      (epoch, batch_number + 1, args.epoch_size, duration, err))
                batch_number += 1
                i += 1
                train_time += duration
                summary.value.add(tag='loss', simple_value=err)

            # Add validation loss and accuracy to summary
            # pylint: disable=maybe-no-member
            summary.value.add(tag='time/selection', simple_value=selection_time)
            summary_writer.add_summary(summary, step)
    return step


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


# sample people
# given a dataset consturcted by a list of paths
# 这里本质上是从每一个人的图片夹中拿出图片进行组合
def sample_people(dataset, people_per_batch, images_per_person):
    # print('begin sample people')
    # total number of images，每个batch需要抽样的张数
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)  # 总共的dataset大小
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # 挑出选中的类的序号
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        # 选出选中的类中的图片
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        # 最终要挑的图片的数量按照三个约束条件中挑一个
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
    # print('finished sampling')
    # 获取到取样后的图片路径list，还有每个类图片的个数
    return image_paths, num_per_class


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/kiwi_fung/label_web/logs/facenet_cow')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='~/kiwi_fung/label_web/models/facenet_cow')
    '''
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    '''

    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    # default='~/kiwi_fung/label_web/model-20180402-114759.ckpt')

    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')

    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)

    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)

    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=45)

    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=6)

    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)

    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)

    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)

    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')

    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')

    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)

    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)

    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')

    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.)

    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)

    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)

    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)

    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')

    parser.add_argument('--file_exp', type=str,
                        help='The directory of the csv labeled for clustering', default='/home/shikigan/out_res_1')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
