
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # Should print False
x = torch.rand(2, 3)
print("Random tensor:\n", x)
