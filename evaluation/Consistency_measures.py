import numbers
from math import log

def calculate_abic(n, ratio, num_params, calctype):
    """
    ratio: reduct ratio or core ratio
    n: number of samples
    num_params: length of intersection set
    """
    complement = 1 - ratio

    if complement < 0:
        return 'Undefined'

    if complement == 0:
        return float('inf')

    if calctype == 'AIC':
        val = -2 * log(complement, 2) + 2 * num_params
    elif calctype == 'BIC':
        val = -2 * log(complement, 2) + num_params * log(n, 2)
    else:
        raise ValueError('Undefined calculation type')

    return val


def Compute_dissimilarity(original, xai_ratio):
  if xai_ratio > 0 and isinstance(xai_ratio, numbers.Number):
    min_val = min(original/xai_ratio, xai_ratio/original)
  else:
    min_val = 'Undefined'
  return min_val

def get_consistency_val(computed_ratio, dataset_ratio):
  if isinstance(computed_ratio, numbers.Number): 
      val = Compute_dissimilarity(dataset_ratio, computed_ratio)
  else:
      val = 'Undefined'  
  return val
  

