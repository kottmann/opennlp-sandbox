
import tensorflow as tf

# TODO: This should be run once during every build, to create graph definition file

#
# Define the placeholders
#

# TODO: Default should be 0 <- disabled, can then be set for training
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prop")

# shape is batch_size, and length of name
char_ids_ph = tf.placeholder(tf.int32, shape=[None, None], name="char_ids")

# shape is batch_size
name_lengths_ph = tf.placeholder(tf.int32, shape=[None], name="name_lengths")

# shape is batch_size
y_ph = tf.placeholder(tf.int32, shape=[None], name="category")

#
# Define the graph
#

# Don't hard code these ... maybe it is better to use variables, and set them during training

nchars = 157

nclasses = 2

dim_char = 100

K = tf.get_variable(name="char_embeddings", dtype=tf.float32,
                    shape=[nchars, dim_char])

char_embeddings = tf.nn.embedding_lookup(K, char_ids_ph)

char_embeddings = tf.nn.dropout(char_embeddings, dropout_keep_prob)

char_hidden_size = 256
cell_fw = tf.contrib.rnn.LSTMCell(char_hidden_size, state_is_tuple=True)
cell_bw = tf.contrib.rnn.LSTMCell(char_hidden_size, state_is_tuple=True)

_, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                      cell_bw,
                                                                      char_embeddings,
                                                                      sequence_length=name_lengths_ph,
                                                                      dtype=tf.float32)

output = tf.concat([output_fw, output_bw], axis=-1)

output = tf.nn.dropout(output, dropout_keep_prob)

W = tf.get_variable("W", shape=[2*char_hidden_size, nclasses])
b = tf.get_variable("b", shape=[nclasses])
logits = tf.nn.xw_plus_b(output, W, b, name="logits")

# softmax ...
probs = tf.exp(logits)
norm_probs = tf.identity(probs / tf.reduce_sum(probs, 1, keepdims=True), name="norm_probs")

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_ph)
mean_loss = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss, name="train")

model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)

builder = tf.saved_model.builder.SavedModelBuilder("./namecat_graph")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING],)
builder.save()

saver = tf.train.Saver()
saver_def = saver.as_saver_def()

# The name of the tensor you must feed with a filename when saving/restoring.
print(saver_def.filename_tensor_name)

# The name of the target operation you must run when restoring.
print(saver_def.restore_op_name)

# The name of the target operation you must run when saving.
print(saver_def.save_tensor_name)

tf.matmul()

# C++ code to save the graph, only need to run one op, and pass it the filename as tensor
#tf::Tensor string( tf::DT_STRING, tf::TensorShape({ 1, 1 } ) );

#Feeding string: string.matrix< std::string >()( 0, 0 ) = file_path_ + filename;
#Execution: TF_CHECK_OK(
#    session_->Run( { { "save/Const:0", string } }, {}, { "save/control_dependency" }, nullptr ) );

# feed dict with "save/Const:0", "/home/blue/saveHERE"
# opt that has to be run "save/control_dependency"
