#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'findNumber' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER_ARRAY arr
#  2. INTEGER k
#

def findNumber(list,k):
    if k in list:
        print('yes')
    else:
        print('no')
    return k

list = [1,2,3,4,6]
k = int(2)
print(findNumber(list,k))