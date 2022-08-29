#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'oddNumbers' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER l
#  2. INTEGER r
#

def oddNumbers(l, r):
    # Write your code here
    for i in range(l, r):
        if i % 2 != 0:
            print(i)
        else:
            print('no')
print(oddNumbers(2, 10))
