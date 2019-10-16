import tensorflow as tf
from collections import Counter, OrderedDict
import pandas as pd
import nltk
import os


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "path to csv containing data")

flags.DEFINE_integer("vocab_size", 30000,
                    "size of vocab file")

flags.DEFINE_integer("max_seq_len", 256, "Maximum sequence length.")

flags.DEFINE_float("train_prop", 0.8, "proportion of data to use for train")

flags.DEFINE_string("vocab_dir", None, "output directory for vocab")

flags.DEFINE_string("output_records_dir", None, "directory in which to store train and test tfrecord files")

def create_vocab(corpus, vocab_size= 30000):

    tokens = []
    
    vocab =  {
       
        "<START>": 0,
        "<END>" : 1,
        "<PAD>" : 2,
        "<UNK>" : 3
    }

    for doc in corpus:
        tokens.extend(map(lambda x:x.lower(),nltk.word_tokenize(doc)))
    
    count_dict = Counter(tokens)
    
    most_common = count_dict.most_common(vocab_size)

    for index,token in enumerate(most_common):
        vocab[token[0]] = index+4

    return vocab


def encode_text(text,vocab,max_seq_len = 256):

    tokens = [token.lower() for token in nltk.word_tokenize(text)]
    encoded_text = [vocab["<START>"]]

    if(len(tokens)<max_seq_len-2):

        encoded_text.extend([vocab.get(i,vocab["<UNK>"]) for i in tokens])
        encoded_text.append(vocab["<END>"])
        encoded_text.extend([vocab["<PAD>"] for i in range(max_seq_len-len(tokens)-2)])
       
    else:

        encoded_text.extend([vocab.get(token,vocab["<UNK>"]) for token in tokens[:max_seq_len-2]])
        encoded_text.append(vocab["<END>"])

    return encoded_text

        
def create_tf_example(text, vocab, max_seq_len=256, label=None):

    input_ids = encode_text(text, vocab, max_seq_len)

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    features = OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["label"] = create_int_feature(label)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    return tf_example.SerializeToString()


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    data = pd.read_csv(FLAGS.input_file)
    data = data.sample(frac=1).reset_index(drop=True)

    train_data = data.loc[:int(len(data)*FLAGS.train_prop), :]
    test_data = data.loc[int(len(data)*FLAGS.train_prop):, :].reset_index(drop=True)

    vocab = create_vocab(train_data["comment_text"])

    vocab_file = open(FLAGS.vocab_dir,"w")
    vocab_file.writelines([token+"\n" for token in vocab.keys()])
    vocab_file.close()


    writer_train = tf.python_io.TFRecordWriter(os.path.join(os.path.normpath(FLAGS.output_records_dir),"train.tfrecords"))
    writer_test = tf.python_io.TFRecordWriter(os.path.join(os.path.normpath(FLAGS.output_records_dir),"test.tfrecords"))

    for index in range(len(train_data)):

        if index%100 ==0:
            print("writing training record no {}".format(index))

        text = train_data["comment_text"][index]
        label = [train_data[column][index] for column in data.columns[-6:]]
        
        tf_example = create_tf_example(text, vocab, FLAGS.max_seq_len,label)
        writer_train.write(tf_example)  

    for index in range(len(test_data)):

        if index%100 ==0:
            print("writing test record no {}".format(index))

        text = test_data["comment_text"][index]
        label = [test_data[column][index] for column in data.columns[-6:]]
        
        tf_example = create_tf_example(text, vocab, FLAGS.max_seq_len,label)
        writer_test.write(tf_example)    


    writer_train.close()
    writer_test.close()
    

 
if __name__ == "__main__":

    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_dir")
    flags.mark_flag_as_required("output_records_dir")
    tf.app.run()

























































































