import numpy as np
import scipy.weave as weave
from scipy.weave import converters

def fastAbsArgMax( array ):
  assert( len(array.shape) == 2 )
  maxIndex = np.zeros( 2 )
  nx, ny = array.shape
  code = \
  """
  double max = 0.0;
  double *value;
  for(int i = 0; i < nx; i++)
  {
    value = array + i*ny + 1;
    for(int j = 0; j < ny; j++)
    {
      if ((*value > max) || (-(*value) > max))
      {
        max = *value > 0 ? *value : -(*value);
        *maxIndex = i;
        *(maxIndex + 1) = j + 1;
      }
      value++;
    }
  }
  """
  weave.inline(code, ['maxIndex', 'array', 'nx', 'ny'], compiler = 'gcc')
  return np.array( maxIndex, dtype=np.int )