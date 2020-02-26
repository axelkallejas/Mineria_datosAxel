#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download('stopwords')


# In[3]:


#nltk.download()
#nltk.download('punkt')


# In[4]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import numpy as np    
from numpy import dot
from numpy.linalg import norm


# In[5]:


stopwords.fileids()


# In[7]:


set(stopwords.words('english'))


# In[8]:


set(stopwords.words('spanish'))


# In[9]:


tokenizer = RegexpTokenizer(r'\w+')


# In[10]:


#Párrafo 1
text_01 = "Se dispara el coronavirus: la cifra de muertos aumenta 242 en la provincia de Hubei, mientras que la cifra mundial de muertes por coronavirus aumenta a 1.357"


# In[11]:


#Párrafo 2
text_02 ="The 2020 #ChineseGP will be postponed due to the coronavirus outbreak F1 and the FIA have accepted a request from the promoter to postpone the event We will continue to monitor the situation and assess potential alternative dates"


# In[19]:


#Párrafo 3
text_03 ="Crea @UNAM_MX Licenciatura en Ingeniería Aeroespacial Se impartirá en la Facultad de Ingeniería, en Ciudad Universitaria, donde los alumnos deberán cursar 10 semestres y cubrir 450 créditos."


# In[13]:


#Párrafo 4
text_04 ="Samsung Galaxy Z Flip cuenta con una pantalla protegida por vidrio ultra fino, tiene una pequeña pantalla en el exterior que muestra la hora o quién te llama y lo más importante: permite hacer selfies sin tocar ningún botón."


# In[14]:


#Parrafo 5
text_05 ="Llegó el 2020 y nos encontramos con una serie de estrenos que nos harán iniciar a lo grande. Sin lugar a dudas, el regreso de Cloud, Tifa y Aerith en Final Fantasy VII Remake, no se trata solamente de uno de los juegos más esperados del año, sino más bien de la década"


# In[ ]:


TOKENIZAR


# In[15]:


text_01_tokens = tokenizer.tokenize(text_01.lower()) #tokenizar y quitar signos de puntuación
#print(text_01_tokens)

text_01_tokens_wout_stopwords = []

for word in text_01_tokens:
    if word not in stopwords.words('spanish'): text_01_tokens_wout_stopwords.append(word)

print(text_01_tokens_wout_stopwords)


# In[17]:


text_02_tokens = tokenizer.tokenize(text_02.lower()) #tokenizar y quitar signos de puntuación
#print(text_02_tokens)

text_02_tokens_wout_stopwords = []

for word in text_02_tokens:
    if word not in stopwords.words('english'): text_02_tokens_wout_stopwords.append(word)

print(text_02_tokens_wout_stopwords)


# In[20]:


text_03_tokens = tokenizer.tokenize(text_03.lower()) 
#print(text_03_tokens)

text_03_tokens_wout_stopwords = []

for word in text_03_tokens:
    if word not in stopwords.words('spanish'): text_03_tokens_wout_stopwords.append(word)

print(text_03_tokens_wout_stopwords)


# In[21]:


text_04_tokens = tokenizer.tokenize(text_03.lower()) 
#print(text_04_tokens)

text_04_tokens_wout_stopwords = []

for word in text_04_tokens:
    if word not in stopwords.words('spanish'): text_04_tokens_wout_stopwords.append(word)

print(text_04_tokens_wout_stopwords)


# In[22]:


text_05_tokens = tokenizer.tokenize(text_05.lower()) 
#print(text_05_tokens)

text_05_tokens_wout_stopwords = []

for word in text_05_tokens:
    if word not in stopwords.words('spanish'): text_05_tokens_wout_stopwords.append(word)

print(text_05_tokens_wout_stopwords)


# In[23]:


print(len(text_01_tokens_wout_stopwords))
print(len(text_02_tokens_wout_stopwords))
print(len(text_03_tokens_wout_stopwords))
print(len(text_04_tokens_wout_stopwords))
print(len(text_05_tokens_wout_stopwords))
print(len(text_01_tokens_wout_stopwords) + len(text_02_tokens_wout_stopwords) + len(text_03_tokens_wout_stopwords)+len(text_04_tokens_wout_stopwords)+len(text_05_tokens_wout_stopwords))


# BoW

# In[24]:


dicc_texts = {"text_01": text_01_tokens_wout_stopwords, 
              "text_02": text_02_tokens_wout_stopwords, 
              "text_03": text_03_tokens_wout_stopwords,
              "text_04": text_04_tokens_wout_stopwords,
              "text_05": text_05_tokens_wout_stopwords,}
#dicc_texts


# In[25]:


dicc_termns = {}

for text in dicc_texts:
    for word in dicc_texts[text]:
        
#        print("EVALUAR:", word, "EN", text)
        
        if(word in dicc_termns):#incrementar palabras al diccionario
            dicc_termns[word] = dicc_termns[word] + 1
            
#            print(word, "IN", "dicc_termns")
            
        elif(word not in dicc_termns):#agregar palabras al diccionario        
            dicc_termns[word] = 1
            
#            print(word, "NOT IN", "dicc_termns")            

print(len(dicc_termns))
dicc_termns

matriz termino documento (binaria)
# In[26]:


matrix = np.zeros((len(dicc_texts), len(dicc_termns))) # Pre-allocate matrix
#matrix


# In[27]:


i = 0
j = 0

for word_termns in dicc_termns: #dicc_termns todos los términos
#    print()
    for word_texts in dicc_texts: #dicc_texts todos los textos
#        print("EVALUAR:", word_termns, "EN: ", word_texts)
        if(word_termns in dicc_texts[word_texts]): #si está
            print(word_termns, "IN", word_texts)
            
            matrix[j, i] = 1
            
        elif(word_termns not in dicc_texts[word_texts]): # si no está
            print(word_termns, "NOT IN", word_texts)
            
            matrix[j, i] = 0
            
            
        print("se agregó: ", matrix[j,i], "en: ", j, i)
            
        j = j + 1
        
    j = 0
    i = i + 1


# In[28]:


matrix


# In[29]:


matrix.shape


# In[30]:


matrix[0]


# In[31]:


matrix[1]


# In[32]:


matrix[2]


# In[33]:


matrix[3]


# In[34]:


matrix[4]


# In[35]:


matrix[5]


# In[36]:


bin_cos_t01_t02 = dot(matrix[0],matrix[1])/(norm(matrix[0])*norm(matrix[1]))
bin_cos_t01_t02


# In[37]:


bin_cos_t01_t03 = dot(matrix[0],matrix[2])/(norm(matrix[0])*norm(matrix[2]))
bin_cos_t01_t03


# In[38]:


bin_cos_t01_t04 = dot(matrix[0],matrix[3])/(norm(matrix[0])*norm(matrix[3]))
bin_cos_t01_t04


# In[39]:


bin_cos_t01_t05 = dot(matrix[0],matrix[4])/(norm(matrix[0])*norm(matrix[4]))
bin_cos_t01_t05


# In[40]:


bin_cos_t02_t03 = dot(matrix[1],matrix[2])/(norm(matrix[1])*norm(matrix[2]))
bin_cos_t02_t03


# In[41]:


bin_cos_t02_t04 = dot(matrix[1],matrix[3])/(norm(matrix[1])*norm(matrix[3]))
bin_cos_t02_t04


# In[42]:


bin_cos_t02_t05 = dot(matrix[1],matrix[4])/(norm(matrix[1])*norm(matrix[4]))
bin_cos_t02_t05


# In[43]:


bin_cos_t03_t04 = dot(matrix[2],matrix[3])/(norm(matrix[2])*norm(matrix[3]))
bin_cos_t03_t04


# In[44]:


bin_cos_t03_t05 = dot(matrix[2],matrix[4])/(norm(matrix[2])*norm(matrix[4]))
bin_cos_t03_t05


# In[45]:


bin_cos_t04_t05 = dot(matrix[3],matrix[4])/(norm(matrix[3])*norm(matrix[4]))
bin_cos_t04_t05


# Matriz termino documento con frecuencia

# In[46]:


matrix = np.zeros((len(dicc_texts), len(dicc_termns))) # Pre-allocate matrix
#matrix


# In[47]:


i = 0
j = 0

for word_termns in dicc_termns: #dicc_termns todos los términos
#    print()
    for word_texts in dicc_texts: #dicc_texts todos los textos
#        print("EVALUAR:", word_termns, "EN: ", word_texts)
        if(word_termns in dicc_texts[word_texts]): #si está
            print(word_termns, "IN", word_texts)
            
            matrix[j, i] = dicc_termns[word_termns]
            
        elif(word_termns not in dicc_texts[word_texts]): # si no está
            print(word_termns, "NOT IN", word_texts)
            
            matrix[j, i] = 0
            
            
        print("se agregó: ", matrix[j,i], "en: ", j, i)
            
        j = j + 1
        
    j = 0
    i = i + 1


# In[48]:


matrix


# In[49]:


matrix.shape


# In[50]:


matrix[0]


# In[51]:


matrix[1]


# In[52]:


matrix[2]


# In[53]:


matrix[3]


# In[54]:


matrix[4]


# In[55]:


matrix[5]


# In[56]:


df_cos_t01_t02 = dot(matrix[0],matrix[1])/(norm(matrix[0])*norm(matrix[1]))
df_cos_t01_t02


# In[57]:


df_cos_t01_t03 = dot(matrix[0],matrix[2])/(norm(matrix[0])*norm(matrix[2]))
df_cos_t01_t03


# In[58]:


df_cos_t01_t04 = dot(matrix[0],matrix[3])/(norm(matrix[0])*norm(matrix[3]))
df_cos_t01_t04


# In[59]:


df_cos_t01_t05 = dot(matrix[0],matrix[4])/(norm(matrix[0])*norm(matrix[4]))
df_cos_t01_t05


# In[60]:


df_cos_t02_t03 = dot(matrix[1],matrix[2])/(norm(matrix[1])*norm(matrix[2]))
df_cos_t02_t03


# In[61]:


df_cos_t02_t04 = dot(matrix[1],matrix[3])/(norm(matrix[1])*norm(matrix[3]))
df_cos_t02_t04


# In[62]:


df_cos_t02_t05 = dot(matrix[1],matrix[4])/(norm(matrix[1])*norm(matrix[4]))
df_cos_t02_t05


# In[63]:


df_cos_t03_t04 = dot(matrix[2],matrix[3])/(norm(matrix[2])*norm(matrix[3]))
df_cos_t03_t04


# In[64]:


df_cos_t03_t05 = dot(matrix[2],matrix[4])/(norm(matrix[2])*norm(matrix[4]))
df_cos_t03_t05


# In[65]:


df_cos_t04_t05 = dot(matrix[3],matrix[4])/(norm(matrix[3])*norm(matrix[4]))
df_cos_t04_t05


# In[ ]:




