from torch import device, cuda

delta = 8
epochs = 150
bsize = 32
freq_printing = 10
device = device("cuda" if cuda.is_available() else "cpu")
