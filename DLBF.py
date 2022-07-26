# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:07:42 2022

@author: admin
"""
import Bloom_filters as BF
import Adaptive_Bloom_Filters as ABF
import pandas as pd
import numpy as np
import tensorflow as tf




class DLBF:
    def  __init__(self):
        """set variables as placeholder for initial and backup filter"""
        self.initial=''
        self.backup=''
        self.model=''
        self.defensive=''         # level of defencing
        self.cutoff={}

    def construct(self, dataset,popular_data,label_column_name, url_column_name,model,model_dataset, memory_size,backup_size,thresholds,seg,defensive_size,alpha,sandwich=True):

        """Initialize all the variables"""
        try:
            all_pos_data=np.array(dataset[dataset[label_column_name]==1][url_column_name])
        except:
            return "Need a ground truth column set to 1/0"

        all_data=np.array(dataset[url_column_name])
        label=np.array(dataset[label_column_name])
        dataset['index']=dataset.index

        if sandwich:
            """Initial Filter"""
            initial_bf=BF.Bloom_Filter(memory_size-backup_size,len(all_pos_data))
            for i in range(len(all_pos_data)):
                initial_bf.insert(all_pos_data[i])

            #run the data through initial bloom filter
            round1_pos=[]
            for i in range(len(all_data)):
                if initial_bf.search(all_data[i])==True:
                    round1_pos.append(i)

            self.initial=initial_bf

            backup_bf=ABF.Bloom_Filter(backup_size,len(round1_pos),seg)

        else:

            round1_pos=np.arange(len(all_data))
            initial_err=0
            backup_bf=ABF.Bloom_Filter(backup_size-defensive_size,len(round1_pos),seg)


        """creating data frame for backup adaptive filter"""
        round1_pos_df=dataset.iloc[round1_pos,:]

        """Classifier"""
        self.model=model
        #self.threshold=threshold
        round1_pos_df=model_dataset.iloc[round1_pos,:].astype(np.float32)#.reshape((1,model_dataset.shape[1]-1))
        
        # interpreter = tf.lite.Interpreter(model_content=model)
        # interpreter.allocate_tensors()
        # self.model=interpreter
        #
        # model_output=[]
        # for i in round1_pos:
        #     interpreter.set_tensor(interpreter.get_input_details()[0]['index'],np.array(model_dataset.iloc[i,:-1]).astype(np.float32).reshape((1,model_dataset.shape[1]-1)))
        #     interpreter.invoke()
        #     model_output.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0])
        model_output=self.model.predict_proba(model_dataset.iloc[round1_pos,:-1].astype(np.float32))[:,-1]
       
        model_popular_output = self.model.predict_proba(popular_data.iloc[:,:-1].drop(columns = 'url')).astype(np.float32)[:,-1]  #get the score of popular url
        #print(model_popular_output)
        round1_pos_df['Prob']=model_output

        round1_pos_df=round1_pos_df.sort_values(by='Prob').reset_index()
        #print(round1_pos_df)

        """Backup ABF"""
        #calculate total c after going throuhgh n number of segments
        
            #print(total_c)
        m = [0,0,0,0,0,0]
        for i in range(len(round1_pos_df)):
            if round1_pos_df.loc[i,'Prob']<thresholds[0]:
                m[1] = i
            elif round1_pos_df.loc[i,'Prob']<thresholds[1]:
                m[2] = i
            elif round1_pos_df.loc[i,'Prob']<thresholds[2]:
                m[3] = i
            elif round1_pos_df.loc[i,'Prob']<thresholds[3]:
                m[4] = i
            else:
                m[5] = i
        #calculate the initial m: the number of points in the first segment, the segment with lowest prob

        round1_pos_df['Segments'] = 0
        popular_data['Segments'] = 0
        #for loop to assign each data point with a segment
        for j in range(backup_bf.segments):
            ms = m[j]
            me = m[j+1]
            if j == (backup_bf.segments-1):
                round1_pos_df.loc[m[-2]:,'Segments']=j
                self.cutoff[round1_pos_df.loc[m[-2],'Prob']]=j
            else:
                round1_pos_df.loc[ms:me,'Segments']=j
                self.cutoff[round1_pos_df.loc[ms,'Prob']]=j
        
        #print(round1_pos_df)
        print(self.cutoff.keys())
        #print(list(self.cutoff.keys()))
        
        cutoff_value = list(self.cutoff.keys())
        
        '''for i in range(self.cutoff):
            cutoff_value.append(list(self.cutoff.keys())[i])'''
        
        
        for k in range(len(model_popular_output)):
            for l in range(len(cutoff_value)-1):
                if model_popular_output[k]>=cutoff_value[l] and model_popular_output[k]<cutoff_value[l+1]:
                    popular_data.loc[k,'Segments'] = l
                if model_popular_output[k] >= cutoff_value[-1]:
                    popular_data.loc[k,'Segments'] = seg-1
        #print(popular_data)
        concrete_num_seg = []
        
        for i in range(backup_bf.segments):
            concrete_num = 0
            for j in range(len(round1_pos_df)):
                if round1_pos_df.loc[j,'Segments']==i:
                    concrete_num += 1
            concrete_num_seg.append(concrete_num)

        positive_array=round1_pos_df[round1_pos_df['Segments']==(backup_bf.segments-1)]['index'].tolist()

        backup_bf_df=round1_pos_df[round1_pos_df['Segments']!=(backup_bf.segments-1)]

    
        pos_backup_bf_df=backup_bf_df[backup_bf_df[label_column_name]==1]
        
        backup_bf_df.set_index('index',inplace=True)
        pos_backup_bf_df.set_index('index',inplace=True)
        
        for i in pos_backup_bf_df.index.tolist():
            backup_bf.insert(all_data[i],pos_backup_bf_df.loc[i,'Segments'])

        #Checking all backup dataset using the ABF
        backup_bf_result=[]
        backup_bf_truth=[]
        for i in backup_bf_df.index.tolist():
            backup_bf_result.append(backup_bf.search(all_data[i],backup_bf_df.loc[i,'Segments']))
            backup_bf_truth.append(int(label[i]))
        
        #dataset = dataset.drop(columns = 'index')
        self.backup=backup_bf
        
        #the following is the part of defensive filter
        '''defensive_filt = []
        for i in range(seg+1):
            defensive_filt.append(BF.Bloom_Filter(defensive_size,len(popular_data)))
        for j in range(len(popular_data)):
            defensive_filt[popular_data.loc[j,'Segments']].insert(popular_data.loc[j,'url'])'''
        #popular_data = popular_data.drop(columns = 'url')
        num_defensive = int(defensive_size/alpha)
        self.defensive = []
        for r in range(seg):
            self.defensive.append([])
        for i in range(num_defensive):
            
            if self.backup.search(popular_data.loc[i,'url'],popular_data.loc[i,'Segments']) == True:
                self.defensive[popular_data.loc[i,'Segments']].append(popular_data.loc[i,'url'])
        #print(popular_data.loc[:,'Segments'])  
        #print('!!!!!',self.cutoff.keys())
        #print(self.defensive)
            
    def search(self,data,model_data):
        if self.initial=='':
            
            # self.model.set_tensor(self.model.get_input_details()[0]['index'],np.array(model_data[:-1]).astype(np.float32).reshape((1,len(model_data)-1)))
            # self.model.invoke()
            prob=self.model.predict_proba(model_data[:-1].values.reshape(1,-1))[-1][-1]
            for i in range(len(list(self.cutoff.keys()))):
                
                if i < self.backup.segments-1:
                    if prob>=list(self.cutoff.keys())[i] and prob<list(self.cutoff.keys())[i+1]:
                        #print(i)
                        segment=self.cutoff[list(self.cutoff.keys())[i]]
                        #print(segment)
                        if segment==self.backup.segments-1:     #the rightmost region 
                            print("case 1: the rightmost region" )
                            if data in self.defensive[-1]:
                                return False
                            else:
                                print('1')
                                return True
                           
                            
                        else:
                            if self.backup.search(data,segment):     #other regions
                                print("case 2: other regions but incur fp")
                                if data in self.defensive[segment]:
                                    return False
                                else:
                                #print(i)
                                    print('2')
                                    return True
                            else:
                                return False
                                
                elif i == self.backup.segments-1:
                    
                    if prob >=list(self.cutoff.keys())[i]:
                        print("case 3")
                        if data in self.defensive[-1]:
                            return False
                        else:
                            print('3')
                            return True
                            
        else:
            if self.initial.search(data):
                # self.model.set_tensor(self.model.get_input_details()[0]['index'],np.array(model_data[:-1]).astype(np.float32).reshape((1,len(model_data)-1)))
                # self.model.invoke()
                prob=self.model.predict_proba(model_data[:-1].values.reshape(1,-1))[-1][-1]
                for i in self.cutoff:
                    if prob>i:
                        segment=self.cutoff[i]
                        if segment==self.backup.segments-1:
                            print('4')
                            return True
                            
                        else:
                            if self.backup.search(data,segment):
                                print('5')
                                return True
                                
                            else:
                                return False
            else:
                return False

