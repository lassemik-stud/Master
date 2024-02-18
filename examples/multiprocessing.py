#!/usr/bin/python3
# encoding: utf-8

import multiprocessing as mp
import numpy as np

def task(item):
    return item % 10

def mp_pool_test():
    pool = mp.Pool(processes=1)
    inputs = range(1000)
    results = pool.map(task, inputs)
    
