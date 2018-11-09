from settings import MODEL_DIR

SEP = '_'

def make_name(lr, momentum, epoch, batch_size, name):
    path = MODEL_DIR + name + SEP + str(lr) + SEP + str(momentum) + SEP + str(epoch) + SEP + str(batch_size)
    return path
