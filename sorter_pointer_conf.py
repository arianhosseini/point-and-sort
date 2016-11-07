from blocks.bricks import Tanh
from blocks.algorithms import BasicMomentum, AdaDelta, RMSProp, Adam, CompositeRule, StepClipping, Momentum
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal
from sorter_pointer import Model

example_count = 1000000
batch_size = 10
sort_batch_count = 20

embed_size = 50
match_skip_connections = True

pre_lstm_size = 100
pre_skip_connections = False

lstm_size = [100]
skip_connections = True

#ptr_net decoder config:
decoder_data_dim =  2*lstm_size[0]
decoder_lstm_output_dim = 2*lstm_size[0]


attention_mlp_hidden = [150]
attention_mlp_activations = [Tanh()]

step_rule = CompositeRule([RMSProp(decay_rate=0.95, learning_rate=5e-5),
                           BasicMomentum(momentum=0.9)])

dropout = 0.2
w_noise = 0.

valid_freq = 10000
save_freq = 10000
print_freq = 1000

weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.)

transition_weights_init = Orthogonal()
