import secrets
import math
import numpy as np

def uniform(a:float=0.0, b:float=1.0) -> float:
  """Cryptographically secure uniform sample."""
  # 53 random bits gives 53-bit precision double
  # Given by Dr. Rahul Bhadani
  u = secrets.randbits(53) / (1 << 53)   # in [0, 1)
  return a + (b - a) * u

def exponentialdist(l:float) -> float:
  """
  Generates a random sample from exponential distribution using inverse transform sampling

  Formula: x = ((-1/l)*np.log(y)) 

  Args:
	l (float): float value to act as lambda in the exponential distribution

  Returns:
  	x (float): float value of the randomly sampled value from exponential distribution

  """

  y= uniform()
  x = ((-1/l)*np.log(y))
  return x


def poissondist(l:float) -> float:
 """
  Generates a random sample from poisson distribution using inverse transform sampling

  Formula: x = (np.exp(-l)*(l**i))/math.factorial(i)

  Args:
        l (float): float value to act as lambda in the poisson distribution

  Returns:
        x (float): float value of the randomly sampled value from poisson distribution

  """

 y = uniform()
 x = 0
 k = 100000
 for i in range(0,k):
  x += (np.exp(-l)*(l**i))/math.factorial(i)
  if x >= y:
    return i


if __name__ == "__main__":
    # Example Usage
    print(f"Uniform Random Sample {uniform()}")
    print(f"Exponential Random Sample {exponentialdist(5)}")
    print(f"Poisson Random Sample {poissondist(5)}")
