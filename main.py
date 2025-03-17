#!/usr/bin/env python
import sys
from src.denovo import main

import torch
import numpy.core.multiarray
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtypes.Float64DType, np.dtype])

if __name__ == "__main__":
    sys.exit(main())