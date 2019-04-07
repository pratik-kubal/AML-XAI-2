#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.inference import VariableElimination
from tqdm import tqdm


# In[2]:


feature_data = pd.read_csv("../dataset/15features.csv")


# In[3]:


for idx,columns in enumerate(feature_data.columns):
    if columns != "imagename":
        feature_data[str(columns)] = feature_data[str(columns)] - 1


# In[4]:


feature_data.head()


# In[5]:


seen_train = pd.read_csv("../dataset/seen-dataset/dataset_seen_training_siamese.csv")
seen_train.head()


# In[6]:


trainData = pd.merge(seen_train,feature_data.add_suffix('1'),left_on="left",right_on="imagename1",how="inner")
trainData = pd.merge(trainData,feature_data.add_suffix('2'),left_on="right",right_on="imagename2",how="inner")
trainData = trainData.drop(["Unnamed: 0","imagename1","imagename2"],axis=1)


# In[7]:


trainData.head()


# In[8]:


combined_model = BayesianModel([('pen_pressure1','is_pen_pressure_sim'),
                                ('pen_pressure2','is_pen_pressure_sim'),
                                ('slantness1','is_slantness_sim'),
                                ('slantness2','is_slantness_sim'),
                                ('tilt1','is_tilt_sim'),
                                ('tilt2','is_tilt_sim'),
                                ('is_slantness_sim','is_tilt_sim'),
                                ('staff_of_a1','is_staff_of_a_sim'),
                                ('staff_of_a2','is_staff_of_a_sim'),
                                ('staff_of_d1','is_staff_of_d_sim'),
                                ('staff_of_d2','is_staff_of_d_sim'),
                                ('is_staff_of_a_sim','is_staff_of_d_sim'),
                                ('entry_stroke_a1','entry_stroke_a_sim'),
                                ('entry_stroke_a2','entry_stroke_a_sim'),
                                ('exit_stroke_d1','is_exit_stroke_d_sim'),
                                ('exit_stroke_d2','is_exit_stroke_d_sim'),
                                ('entry_stroke_a_sim','is_exit_stroke_d_sim'),
                                ('is_lowercase1','is_lowercase_sim'),
                                ('is_lowercase2','is_lowercase_sim'),
                                ('is_continuous1','is_continuous_sim'),
                                ('is_continuous2','is_continuous_sim'),
                                ('is_lowercase_sim','is_continuous_sim'),
                                ('dimension1','dimension_sim'),
                                ('dimension2','dimension_sim'),
                                ('letter_spacing1','letter_spacing_sim'),
                                ('letter_spacing2','letter_spacing_sim'),
                                ('size1','size_sim'),
                                ('size2','size_sim'),
                                ('dimension_sim','size_sim'),
                                ('letter_spacing_sim','size_sim'),
                                ('constancy1','constancy_sim'),
                                ('constancy2','constancy_sim'),
                                ('size_sim','constancy_sim'),
                                ('word_formation1','word_formation_sim'),
                                ('word_formation2','word_formation_sim'),
                                ('constancy_sim','word_formation_sim'),
                                ('formation_n1','formation_n_sim'),
                                ('formation_n2','formation_n_sim'),
                                ('word_formation_sim','formation_n_sim')
                               ])

cpd_pen_pressure1 = TabularCPD('pen_pressure1',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_pen_pressure2 = TabularCPD('pen_pressure2',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_is_pen_pressure_sim = TabularCPD('is_pen_pressure_sim',2,[[0.1,0.9,0.9,0.1],
                                                            [0.9,0.1,0.1,0.9]],
                                                            evidence=['pen_pressure1','pen_pressure2'], 
                                                            evidence_card=[2,2])
cpd_slantness1 = TabularCPD('slantness1',4,[[0.25],[0.25],[0.25],[0.25]],
                                                evidence=[], evidence_card=[])
cpd_slantness2 = TabularCPD('slantness2',4,[[0.25],[0.25],[0.25],[0.25]],
                                                evidence=[], evidence_card=[])
cpd_is_slantness_sim = TabularCPD('is_slantness_sim',2,[[0.1,0.2,0.3,0.4,0.2,0.1,0.3,0.4,0.3,0.2,0.1,0.4,0.4,0.3,0.2,0.1],
                                                            [0.9,0.8,0.7,0.6,0.8,0.9,0.7,0.6,0.7,0.8,0.9,0.6,0.6,0.7,0.8,0.9]],
                                                            evidence=['slantness1','slantness2'], 
                                                            evidence_card=[4,4])
cpd_tilt1 = TabularCPD('tilt1',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_tilt2 = TabularCPD('tilt2',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_is_tilt_sim = TabularCPD('is_tilt_sim',2,[[0.4,0.1,0.9,0.6,0.9,0.6,0.4,0.1],
                                                            [0.6,0.9,0.1,0.4,0.1,0.4,0.6,0.9]],
                                                            evidence=['tilt1','tilt2','is_slantness_sim'], 
                                                            evidence_card=[2,2,2])
cpd_staff_of_a1 = TabularCPD('staff_of_a1',4,[[0.25],[0.25],[0.25],[0.25]],
                                                evidence=[], evidence_card=[])
cpd_staff_of_a2 = TabularCPD('staff_of_a2',4,[[0.25],[0.25],[0.25],[0.25]],
                                                evidence=[], evidence_card=[])
cpd_is_staff_of_a_sim = TabularCPD('is_staff_of_a_sim',2,[[0.1,0.2,0.3,0.4,0.2,0.1,0.3,0.4,0.3,0.2,0.1,0.4,0.4,0.3,0.2,0.1],
                                                            [0.9,0.8,0.7,0.6,0.8,0.9,0.7,0.6,0.7,0.8,0.9,0.6,0.6,0.7,0.8,0.9]],
                                                            evidence=['staff_of_a1','staff_of_a2'], 
                                                            evidence_card=[4,4])
cpd_staff_of_d1 = TabularCPD('staff_of_d1',3,[[0.33],
                                    [0.34],[0.33]],
                                    evidence=[], evidence_card=[])
cpd_staff_of_d2 = TabularCPD('staff_of_d2',3,[[0.33],
                                    [0.34],[0.33]],
                                    evidence=[], evidence_card=[])
cpd_is_staff_of_d_sim = TabularCPD('is_staff_of_d_sim',2,[[0.4,0.1,0.9,0.6,0.9,0.6,0.1,0.6,0.4,0.1,0.9,0.6,0.9,0.6,0.9,0.6,0.4,0.9],
                                              [0.6,0.9,0.1,0.4,0.1,0.4,0.9,0.4,0.6,0.9,0.1,0.4,0.1,0.4,0.1,0.4,0.6,0.1]],
                             evidence=['staff_of_d1','staff_of_d2','is_staff_of_a_sim'], 
                             evidence_card=[3,3,2])
cpd_exit_stroke_d1 = TabularCPD('exit_stroke_d1',4,[[0.25],[0.25],[0.25],[0.25]],
                                                evidence=[], evidence_card=[])
cpd_exit_stroke_d2 = TabularCPD('exit_stroke_d2',4,[[0.25],[0.25],[0.25],[0.25]],
                                                evidence=[], evidence_card=[])
cpd_is_exit_stroke_d_sim = TabularCPD('is_exit_stroke_d_sim',2,[[0.9,0.1,0.9,0.6,0.9,0.6,0.9,0.6,0.9,0.6,0.4,0.1,0.9,0.6,0.9,0.6,0.9,0.6,0.9,0.6,0.4,0.1,0.9,0.6,0.9,0.6,0.9,0.6,0.9,0.6,0.4,0.1],
                                                            [0.1,0.9,0.1,0.4,0.1,0.4,0.1,0.4,0.1,0.4,0.6,0.9,0.1,0.4,0.1,0.4,0.1,0.4,0.1,0.4,0.6,0.9,0.1,0.4,0.1,0.4,0.1,0.4,0.1,0.4,0.6,0.9]],
                                                            evidence=['exit_stroke_d1','exit_stroke_d2','entry_stroke_a_sim'], 
                                                            evidence_card=[4,4,2])

cpd_is_lowercase1 = TabularCPD('is_lowercase1',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_is_lowercase2 = TabularCPD('is_lowercase2',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_is_continuous1 = TabularCPD('is_continuous1',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_is_continuous2 = TabularCPD('is_continuous2',2,[[0.5],
                                                [0.5]],
                                                evidence=[], evidence_card=[])
cpd_dimension1 = TabularCPD('dimension1',3,[[0.33],
                                    [0.34],[0.33]],
                                                evidence=[], evidence_card=[])
cpd_dimension2 = TabularCPD('dimension2',3,[[0.33],
                                    [0.34],[0.33]],
                                                evidence=[], evidence_card=[])
cpd_letter_spacing1 = TabularCPD('letter_spacing1',3,[[0.33],
                                    [0.34],[0.33]],
                                    evidence=[], evidence_card=[])
cpd_letter_spacing2 = TabularCPD('letter_spacing2',3,[[0.33],
                                    [0.34],[0.33]],
                                    evidence=[], evidence_card=[])
cpd_size1 = TabularCPD('size1',3,[[0.33],
                                    [0.34],[0.33]],
                                    evidence=[], evidence_card=[])
cpd_size2 = TabularCPD('size2',3,[[0.33],
                                    [0.34],[0.33]],
                                    evidence=[], evidence_card=[])
cpd_constancy1 = TabularCPD('constancy1',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_constancy2 = TabularCPD('constancy2',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_word_formation1 = TabularCPD('word_formation1',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_word_formation2 = TabularCPD('word_formation2',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_formation_n1 = TabularCPD('formation_n1',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_formation_n2 = TabularCPD('formation_n2',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_entry_stroke_a1 = TabularCPD('entry_stroke_a1',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_entry_stroke_a2 = TabularCPD('entry_stroke_a2',2,[[0.5],
                                    [0.5]],
                                    evidence=[], evidence_card=[])
cpd_is_lowercase_sim = TabularCPD('is_lowercase_sim',2,[[0.1,0.9,0.9,0.1],
                                                            [0.9,0.1,0.1,0.9]],
                                                            evidence=['is_lowercase1','is_lowercase2'], 
                                                            evidence_card=[2,2])
cpd_is_continuous_sim = TabularCPD('is_continuous_sim',2,[[0.9,0.1,0.9,0.6,0.9,0.6,0.9,0.1],
                                                            [0.1,0.9,0.1,0.4,0.1,0.4,0.1,0.9]],
                                                            evidence=['is_continuous1','is_continuous2','is_lowercase_sim'], 
                                                            evidence_card=[2,2,2])
cpd_dimension_sim = TabularCPD('dimension_sim',2,[[0.1,0.8,0.9,0.8,0.1,0.8,0.9,0.8,0.1],
                                                [0.9,0.2,0.1,0.2,0.9,0.2,0.1,0.2,0.9]],
                                                evidence=['dimension1','dimension2'], evidence_card=[3,3])
cpd_letter_spacing_sim = TabularCPD('letter_spacing_sim',2,[[0.1,0.8,0.9,0.8,0.1,0.8,0.9,0.8,0.1],
                                                [0.9,0.2,0.1,0.2,0.9,0.2,0.1,0.2,0.9]],
                                                evidence=['letter_spacing1','letter_spacing2'], evidence_card=[3,3])
cpd_size_sim = TabularCPD('size_sim',2,[[0.6,0.3,0.3,0.1,0.8,0.7,0.7,0.3,0.9,0.8,0.7,0.4,0.7,0.6,0.6,0.3,0.6,0.3,0.3,0.1,0.8,0.4,0.4,0.85,0.9,0.8,0.8,0.3,0.8,0.4,0.4,0.85,0.6,0.3,0.3,0.1],
                                        [0.4,0.7,0.7,0.9,0.2,0.3,0.3,0.7,0.1,0.2,0.3,0.6,0.3,0.4,0.4,0.7,0.4,0.7,0.7,0.9,0.2,0.6,0.6,0.15,0.1,0.2,0.2,0.7,0.2,0.6,0.6,0.15,0.4,0.7,0.7,0.9]],
                                        evidence=['size1','size2','dimension_sim','letter_spacing_sim'], evidence_card=[3,3,2,2])
cpd_constancy_sim = TabularCPD('constancy_sim',2,[[0.9,0.1,0.9,0.6,0.9,0.6,0.7,0.1],
                                        [0.1,0.9,0.1,0.4,0.1,0.4,0.3,0.9]],
                                        evidence=['constancy1','constancy2','size_sim'], evidence_card=[2,2,2])
cpd_word_formation_sim = TabularCPD('word_formation_sim',2,[[0.9,0.1,0.9,0.7,0.9,0.7,0.9,0.1],
                                        [0.1,0.9,0.1,0.3,0.1,0.3,0.1,0.9]],
                                        evidence=['word_formation1','word_formation2','constancy_sim'], evidence_card=[2,2,2])
cpd_formation_n_sim = TabularCPD('formation_n_sim',2,[[0.7,0.1,0.9,0.4,0.9,0.4,0.6,0.1],
                                        [0.3,0.9,0.1,0.6,0.1,0.6,0.4,0.9]],
                                        evidence=['formation_n1','formation_n2','word_formation_sim'], evidence_card=[2,2,2])
cpd_entry_stroke_a_sim = TabularCPD('entry_stroke_a_sim',2,[[0.1,0.9,0.9,0.1],
                                                            [0.9,0.1,0.1,0.9]],
                                        evidence=['entry_stroke_a1','entry_stroke_a2'], evidence_card=[2,2])

combined_model.add_cpds(cpd_pen_pressure1,
                        cpd_pen_pressure2,
                        cpd_is_pen_pressure_sim,
                        cpd_slantness1,
                        cpd_slantness2,
                        cpd_is_slantness_sim,
                        cpd_tilt1,
                        cpd_tilt2,
                        cpd_is_tilt_sim,
                        cpd_staff_of_a1,
                        cpd_staff_of_a2,
                        cpd_is_staff_of_a_sim,
                        cpd_staff_of_d1,
                        cpd_staff_of_d2,
                        cpd_is_staff_of_d_sim,
                        cpd_exit_stroke_d1,
                        cpd_exit_stroke_d2,
                        cpd_is_exit_stroke_d_sim,
                        cpd_is_lowercase1,
                        cpd_is_lowercase2,
                        cpd_is_lowercase_sim,
                        cpd_is_continuous1,
                        cpd_is_continuous2,
                        cpd_is_continuous_sim,
                        cpd_dimension1,
                        cpd_dimension2,
                        cpd_dimension_sim,
                        cpd_letter_spacing1,
                        cpd_letter_spacing2,
                        cpd_letter_spacing_sim,
                        cpd_size1,
                        cpd_size2,
                        cpd_size_sim,
                        cpd_constancy1,
                        cpd_constancy2,
                        cpd_constancy_sim,
                        cpd_word_formation1,
                        cpd_word_formation2,
                        cpd_word_formation_sim,
                        cpd_formation_n1,
                        cpd_formation_n2,
                        cpd_formation_n_sim,
                        cpd_entry_stroke_a1,
                        cpd_entry_stroke_a2,
                        cpd_entry_stroke_a_sim
                       )
combined_model.check_model()


# In[9]:


mle = VariableElimination(combined_model)


# In[10]:


for idx,columns in enumerate(feature_data.columns):
    if idx != 0:
        print(str(np.unique(feature_data[columns]))+columns)


# ### Learning the weights in Structured CPD

# In[22]:


simFeatures = [[] for _ in range(len(trainData))]
var = {'is_pen_pressure_sim',
       'is_slantness_sim',
       'is_tilt_sim',
       'is_staff_of_a_sim',
       'is_staff_of_d_sim',
       'entry_stroke_a_sim',
       'is_exit_stroke_d_sim',
      'is_lowercase_sim',
      'is_continuous_sim',
      'dimension_sim',
      'letter_spacing_sim',
       'size_sim',
       'constancy_sim',
       'word_formation_sim',
       'formation_n_sim'
      }
evidence_labels = trainData.columns[3:]
for idx in tqdm(range(len(trainData))):
    inf = mle.query(variables=var,evidence=dict(zip(evidence_labels,trainData.iloc[idx,3:].tolist())))
    for simfeature in var:
        simFeatures[idx].append(np.argmax(inf[simfeature].values))


# In[29]:


simDf = pd.DataFrame(data=simFeatures,columns=var)


# In[30]:


simDf = pd.concat([simDf,trainData.label],axis=1)


# In[31]:


simDf.to_csv("./sigTrainData.csv")


# In[ ]:




