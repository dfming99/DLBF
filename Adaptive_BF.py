


import mmh3
import numpy as np
import math


class Bloom_Filter:
    
    def __init__(self,mem,n,segments):
        self.bit_array_size=mem

        self.length=n

        self.bit_array=np.zeros(self.bit_array_size,dtype=int)

        self.max_hash_num=2*int(math.floor((self.bit_array_size/self.length)*np.log(2)))
        #self.segments=self.max_hash_num
        self.segments=segments
        self.d_dict={segment:0 for segment in range(self.segments)}

        self.fp_rate=0
        self.num_ex = self.segments*0

    
    def insert(self,item,segment_id):
        
        hash_num=self.segments-segment_id
        self.d_dict[segment_id]=self.d_dict[segment_id]+1
        for i in range(hash_num):
            index=mmh3.hash(item,i+self.num_ex)%(self.bit_array_size)
            self.bit_array[index]=1
        return

    
    def search(self,item,segment_id):
        hash_num=self.segments-segment_id-1
        #print(hash_num)
        
        true_hash=0

        for i in range(hash_num):
            index=mmh3.hash(item,i+self.num_ex)%(self.bit_array_size)
            if self.bit_array[index]==1:
                true_hash=true_hash+1
            else:
                None
        #print(true_hash)
        return true_hash==hash_num

    
    def fp_calculator(self):
        fp_val=0
        fp_val_array = []
        for i in range(self.segments):
            fp_val=fp_val+((self.d_dict[i]/self.bit_array_size)*(1-math.exp((-(self.max_hash_num-i)*self.length)/self.bit_array_size))**(self.max_hash_num-i))
            fp_val_array.append(((1-math.exp((-(self.max_hash_num-i)*self.length)/self.bit_array_size))**(self.max_hash_num-i)))
        self.fp_rate=fp_val

        return fp_val ,fp_val_array
