"""This file is regular Bloom filter
"""

"""Use murmur package for hash functions"""
import mmh3

import numpy as np

import math
"""Bloom Filter Class"""
class Bloom_Filter:
    """Inititalization function that takes in the false positive rate
        and number of elements to be inserted"""
    def __init__(self,memory,count):
        self.array_size=memory

        self.length=count

        """Initiate a numpy array with zeros"""
        self.array=np.zeros(self.array_size,dtype=int)

        """calculate the optimal number of hash functions needed"""
        if int(math.floor((self.array_size/self.length)*np.log(2)))!= 0:
            self.hash_num=int(math.floor((self.array_size/self.length)*np.log(2)))
        else:
            self.hash_num = 1

        """Calculate false positive rate"""
        self.false_pos=(1-math.exp((-self.hash_num*self.length)/self.array_size))**self.hash_num

        self.num_ex = self.hash_num*9
    """Insertion method"""
    def insert(self,item):

        """hash the item to be inserted to different hash functions with different seed numbers"""
        for i in range(self.hash_num):

            """The result mod the size of the array to get the index"""
            index=mmh3.hash(item,i+self.num_ex)%(self.array_size)

            """set the specific index to 1"""
            self.array[index]=1
        return

    """Search method"""
    def search(self,item):

        """Variable to indicate if the item is in the array"""
        found=0

        for i in range(self.hash_num):
            index=mmh3.hash(item,i+self.num_ex)%(self.array_size)

            """If all the indices is already 1, then found=True"""
            if self.array[index]==1:
                found=found+1
            else:
                None
        #print(found)
        return found==self.hash_num
'''<<<<<<< HEAD
=======


"""TESTING"""
helper=Helper.Helpers()
helper.read_txt('sample.txt')
bf_test=Bloom_Filter(0.05,helper.word_count)

bf_test=Bloom_Filter(0.2,1500)
bf_test.array_size

for line in helper.words:
    for word in line.split():
        bf_test.insert(word)

>>>>>>> 0488acadc5a6d8891d8198a96f79176e09233cc4'''
