import argparse
import psutil
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Add

from typing import Dict, Any, Callable, Tuple
import numpy as np



flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("train_records_path", None,
                    "path to train.tfrecords")

flags.DEFINE_string("val_records_path", None,
                    "path to test.tfrecords")

flags.DEFINE_integer("epochs", 10, "total train epochs")

flags.DEFINE_integer("batch_size", 32, "training batch size")

flags.DEFINE_string("glove_path", None, "path to glove embeds")

flags.DEFINE_integer("embed_dims", 300, "embedding dimensions")

flags.DEFINE_string("vocab_path", None, "path to vocab file")

flags.DEFINE_string("model_dir", None, "path to store model related files")



def load_embedding_matrix(params):

    
    embedding_dimension = 300
    
    glove_embeds = {}
    embedding_matrix = []

    for line in open(params["glove_path"],"r"):

        glove_embeds[line.split()[0]] = list(map(lambda x:float(x),line.split()[1:]))
       
   
    for word in open(params["vocab_path"],"r"):

        try:
            embedding_matrix.append(np.array(glove_embeds[word]))
        except:
            embedding_matrix.append(np.random.randn(embedding_dimension))

    
    return np.array(embedding_matrix).shape



def data_input_fn(params,is_training=True) -> Callable[[], Tuple]:

    """Return the input function to get the test data.
    Args:
        data_param: data object
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
    Returns:
        Input function:
            - Function that returns (features, labels) when called.
    """
    
    _cpu_core_count = psutil.cpu_count(logical=False)


    def decode_record(record):

        name_to_features = {

            "input_ids": tf.FixedLenFeature([256], tf.int64),
            "label" : tf.FixedLenFeature([6], tf.int64)
                }

        example = tf.parse_single_example(record, name_to_features)

        return example
    
    def _input_fn() -> Tuple:
        
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """  

        if is_training:
            data_path = params["train_records_path"]
        else:
            data_path = params["validation_records_path"]

        dataset = tf.data.TFRecordDataset([data_path])
        dataset = dataset.map(decode_record)

        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat(params["epochs"]) # Infinite iterations: let experiment determine num_epochs
        
        dataset = dataset.batch(params["batch_size"])
        return dataset
    
    return _input_fn



def get_train(params) -> Callable[[], Tuple]:
    """Return training input_fn"""
    return data_input_fn(params,is_training=True)


def get_validation(params) -> Callable[[], Tuple]:
    """Return validation input_fn"""
    return data_input_fn(params, is_training=False)    


def model(features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys, params: Dict[str, Any]):
    
    glove_weights_initializer = tf.constant_initializer(load_embedding_matrix(params))

    print("here are the features")
    print(features)

    embeddings = Embedding(params["vocab_size"], params["embed_dims"], input_length = 256,
                                 embeddings_initializer = glove_weights_initializer)(features["input_ids"])
    
    bidirectional_outputs = Bidirectional(LSTM(128, return_sequences=True))(embeddings)

    pooled_bi_output = GlobalAveragePooling1D()(bidirectional_outputs)

    logits = Dense(6, activation="sigmoid")(pooled_bi_output)

    return logits

def custom_model_fn(features: Dict[str, tf.Tensor],  
                    mode: tf.estimator.ModeKeys, 
                    params: Dict[str, Any]=None) -> tf.estimator.EstimatorSpec:
    
    
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """

    model_output = model(features, mode, params)
    print("eueue")
    
    # Get prediction of model output
    
    predictions = {
            'classes': model_output,
        'probabilities': model_output
    }
    
    # PREDICT

    if mode == tf.estimator.ModeKeys.PREDICT:
        
        export_outputs = {

            'predict_output': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    batch_loss = binary_crossentropy(
        y_true = tf.cast(features["label"], tf.float32),
        y_pred = model_output
    )

    loss = tf.math.reduce_sum(batch_loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        print("fhfhfhf")
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params["learning_rate"],
            optimizer=tf.train.AdamOptimizer(params["learning_rate"])
        )
        
        # Return an EstimatorSpec object for training
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

   
    eval_metric = {
        'accuracy': tf.metrics.accuracy(
            labels=tf.cast(features["label"], tf.int32),
            predictions=model_output,
            name='accuracy'
        )
    }    
    
    # Return a EstimatorSpec object for evaluation
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric)


def main(_):

    
    params = {

        "train_records_path" : FLAGS.train_records_path,
        "val_records_path" : FLAGS.val_records_path,
        "epochs" : FLAGS.epochs,
        "batch_size" : FLAGS.batch_size,
        "glove_path" : FLAGS.glove_path,
        "embed_dims" : FLAGS.embed_dims,
        "vocab_path" : FLAGS.vocab_path,
        "vocab_size" : 30004,
        "learning_rate" : 0.001

    }

    
   # n_training_samples = 0
    
    #for record in tf.python_io.tf_record_iterator(params["train_records_path"]):
     #   n_training_samples+= 1


   # print("number of training examples {}".format(n_training_samples))

   #steps_per_epoch = int(n_training_samples/params["batch_size"])+1
     
   # total_steps = steps_per_epoch*params["epochs"]

    run_config = tf.estimator.RunConfig(model_dir = FLAGS.model_dir, save_checkpoints_steps = 50)

    estimator = tf.estimator.Estimator(model_fn=custom_model_fn, config=run_config, params = params)

    train_input_fn = get_train(params)
    val_input_fn = get_validation(params)

    estimator.train(input_fn=train_input_fn, steps=200)
    

 
if __name__ == "__main__":

    flags.mark_flag_as_required("train_records_path")
    flags.mark_flag_as_required("val_records_path")
    flags.mark_flag_as_required("glove_path")
    flags.mark_flag_as_required("vocab_path")
    flags.mark_flag_as_required("model_dir")

    tf.app.run()
















   


   
   



    
















