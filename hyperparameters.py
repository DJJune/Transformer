from symbols import vocab

epochs = 10
embedding_size = 512
output_size = len(vocab)   # vocab size
num_block = 3              # the number of attention blocks
# embedding_size must be divisible by heads
heads = 4                  # the number of attention heads

lr = 0.001
image_step = 10            # frequency plot the attention map
save_step = 10
checkpoint_path = "./checkpoints"
