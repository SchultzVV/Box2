import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#import timeit
#t0= timeit.timeit()
from mpl_toolkits import mplot3d
import torch.nn as nn
from matplotlib import cm
import pickle
import random as rd
import torch.optim as optim
import numpy as np;import sys as s
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math
#           GERANDO O BANCO DE DADOS

#-----------------------------------------------------------------------
#------------------LOAD DATA-----------------------------------------------------
#-----------------------------------------------------------------------
inp = pickle.load( open( "positions", "rb" ) )
question= pickle.load( open( "question", "rb" ) )
out =  pickle.load( open( "positions", "rb" ) )
Constantes =  pickle.load( open( "Constantes", "rb" ) )
K=Constantes[0];B=Constantes[1]
n_batch=np.shape(inp)[0]
batch_size=np.shape(inp)[1]
n_examples=np.shape(inp)[2]
#inp_test = pickle.load( open( "positions_test", "rb" ) )
#question_test= pickle.load( open( "question_test", "rb" ) )
#out_test =  pickle.load( open( "positions_test", "rb" ) )

#print(n_batch)
#print(batch_size)
#print(n_examples)
#Z=K[0].reshape(500)
#print(np.shape(K[0]))
#print(np.shape(Z))
#print(K[0][0])
#print(Z[0])
#s.exit()
#batch_size=np.shape(inp)
#print(np.shape(question))
#print(np.shape(out))
#-----------------------------------------------------------------------

#------------------ORGANIZE DATA-----------------------------------------------------
#-----------------------------------------------------------------------
#plt.plot(question[0][0].detach().numpy(),out[0][0].detach().numpy())
#plt.show()
#s.exit()
train_loader = torch.utils.data.DataLoader(inp, batch_size=batch_size)
#inp=torch.as_tensor(inp)
#out=torch.as_tensor(out)
#question=torch.as_tensor(question)
#s.exit()
#-------------------------------------------------------------------------------
#------------------DEFINE O MODELO----------------------------------------------
#-------------------------------------------------------------------------------
# Checar os dados dos inputs estão variando e não sendo os mesmos.
# Aumentar o número de neurônios das camadas ocultas.
# Ou mudar para uma arquitetura variacional.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------DEFINE O MODELO----------------------------------------------
#-------------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        # N, 50
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50,400),
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(400,300),
            #nn.ReLU(),
            #nn.ELU(),
            nn.Tanh(),
            nn.Linear(300,200),
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(200,3),
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            #nn.Linear(10,3),
            #nn.ELU(),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Linear(25,10),
        )
        self.project=nn.Linear(1,3)
        self.decoder=nn.Sequential(
            #nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(6,100),
            #nn.ELU(),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(100,200),
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(200,450),
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(450,50),
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(50,1),
            #nn.Tanh()
        )
    def forward(self, x, t):
        encoded = self.encoder(x)
        t=self.project(t)
        aux=torch.cat((encoded,t),1)
        #print(np.shape(aux))
        decoded = self.decoder(aux)
        return decoded,encoded
#-------------------------------------------------------------------------------
PATH='Estado_talvez_funcionante4.pt'  # esse aqui foi treinado por 10k epochs,
                                      #3 Latent e 6 project of dataset(5,500,50)
#PATH='Estado_talvez_funcionante4.pt'
model=Autoencoder()
#torch.save(state, filepath)
model.load_state_dict(torch.load(PATH))
#state = {
#    'epoch': epoch,
#    'state_dict': model.state_dict(),
#    'optimizer': optimizer.state_dict(),
#    ...
#}
#--------------------------------------------------------------------------
#---------------------GRÁFICOS---------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def Predict_test_Scynet():
    t=torch.as_tensor(np.zeros((batch_size,1)))
    t=t.float()
    Y=np.zeros(50);        T=[i for i in range(0,50)]
    #rdn_batch=rd.randint(0,batch_size)
    #aux=[i for i in range(0,n_batch)]
    for aux in range(0,n_batch):
        for rdn_batch in range(0,batch_size):
            YY=inp[aux][rdn_batch].detach().numpy()
            r=0
            for interval in range(0,49):
                for i in range(batch_size):
                    t[i][0]=question[0,i,r]
                y,latent=model(inp[aux].float(),t)
                #y=y.detach.numpy()[rdn_batch]
                y=y.detach().numpy()[rdn_batch]
                Y[interval]=y
                r+=1
            #print(np.shape(YY))
            #print(np.shape(Y))
            #s.exit()
            plt.clf()
            plt.xlim([0, 50])
            plt.ylim([-1, 1])
            plt.plot(T,Y,label='predict',ls='dashed')
            plt.plot(T,YY,label='equation')
            #plt.scatter(T, Y,c='black',label='recon')
            #plt.scatter(T, YY,c='red',label='answ')
            plt.legend()
            plt.pause(0.03)
            #plt.close()
    plt.show()
#Predict_test_Scynet()
#sys.exit()
#--------------------------------------------------------------------------
def Latent_values_Scynet():
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}]],
                        subplot_titles=['Latent Activation 1', 'Latent Activation 2', 'Latent Activation 3'],
                        )
    t=torch.as_tensor(np.zeros((batch_size,1)))
    t=t.float()
    L1,L2,L3=np.zeros(batch_size),np.zeros(batch_size),np.zeros(batch_size)
    ks,bs=np.zeros(batch_size),np.zeros(batch_size)
    Y=np.zeros(50);        T=[i for i in range(0,50)]
    r=25 # tempo escolhido para a pergunta da rede neural
    #ks=[]
    #bs=[]
    for i in range(batch_size):
        t[i][0]=question[0,i,r]
    #rdn_batch=rd.randint(0,batch_size)
    #aux=[i for i in range(0,n_batch)]
    for aux in range(0,1):#n_batch):
        #k=np.array(K[aux]).reshape(batch_size)#[rdn_batch])
        #b=np.array(B[aux]).reshape(batch_size)#[rdn_batch])
        for rdn_batch in range(0,batch_size):
            #YY=inp[aux][rdn_batch].detach().numpy()
            y,latent=model(inp[aux].float(),t)
            #y=y.detach().numpy()[rdn_batch]
            #Y[interval]=y
            L1[rdn_batch] = latent[rdn_batch][0].detach().numpy()
            L2[rdn_batch] = latent[rdn_batch][1].detach().numpy()
            L3[rdn_batch] = latent[rdn_batch][2].detach().numpy()
            ks[rdn_batch] = K[aux][rdn_batch]
            bs[rdn_batch] = B[aux][rdn_batch]
            #ks.append(K[aux][rdn_batch])
            #bs.append(B[aux][rdn_batch])
        fig.add_trace(go.Scatter3d(x=bs,y=ks,z=L1,mode='markers',marker=dict(
            size=12,color=L1,colorscale='Viridis',opacity=0.8)), 1, 1)
        fig.add_trace(go.Scatter3d(x=bs,y=ks,z=L2,mode='markers',marker=dict(
            size=12,color=L2,colorscale='Viridis',opacity=0.8)), 1, 2)
        fig.add_trace(go.Scatter3d(x=bs,y=ks,z=L3,mode='markers',marker=dict(
            size=12,color=L3,colorscale='Viridis',opacity=0.8)), 1, 3)
        #fig = go.Figure(data=[go.Scatter3d(x=bs,y=ks,z=L1,mode='markers',marker=dict(
        #    size=12,color=L1,colorscale='Viridis',opacity=0.8))])
        fig.show()
        #markers=dict(size=12,color=L1,colorscale='Viridis',opacity=0.8)
        #ax1.scatter3D(bs,ks,L1,label='Latent Activation 1',mode='markers')
        #markers=dict(size=12,color=L2,colorscale='Viridis',opacity=0.8)
        #ax2.scatter3D(bs,ks,L2,label='Latent Activation 2',mode='markers')
        #markers=dict(size=12,color=L3,colorscale='Viridis',opacity=0.8)
        #ax3.scatter3D(bs,ks,L3,label='Latent Activation 3',mode='markers')
        #ks=[]
        #bs=[]
    #fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    plt.show()
#plt.pause(10)
        #plt.close()
    #plt.legend()
Latent_values_Scynet()
print('wtf')
s.exit()
#--------------------------------------------------------------------------
def Latent_values_Scynet():
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    t=torch.as_tensor(np.zeros((batch_size,1)))
    t=t.float()
    L1,L2,L3=np.zeros(batch_size),np.zeros(batch_size),np.zeros(batch_size)
    Y=np.zeros(50);        T=[i for i in range(0,50)]
    r=25
    for i in range(batch_size):
        t[i][0]=question[0,i,r]
    #rdn_batch=rd.randint(0,batch_size)
    #aux=[i for i in range(0,n_batch)]
    for aux in range(0,n_batch):
        for rdn_batch in range(0,batch_size):
            #YY=inp[aux][rdn_batch].detach().numpy()
            y,latent=model(inp[aux].float(),t)
            #y=y.detach().numpy()[rdn_batch]
            #Y[interval]=y
            L1[rdn_batch] = latent[rdn_batch][0].detach().numpy()
            L2[rdn_batch] = latent[rdn_batch][1].detach().numpy()
            L3[rdn_batch] = latent[rdn_batch][2].detach().numpy()
            um   = latent[rdn_batch][0].detach().numpy()#.reshape(500)
            #um   = latent[rdn_batch][0].detach().numpy()#.reshape(500)
            dois = latent[rdn_batch][1].detach().numpy()#.reshape(500)
            #dois = latent[rdn_batch][1].detach().numpy()#.reshape(500)
            tres = latent[rdn_batch][2].detach().numpy()#.reshape(500)
            #tres = latent[rdn_batch][2].detach().numpy()#.reshape(500)
            um=np.array(um)
            dois=np.array(dois)
            tres=np.array(tres)
            k=np.array(K[aux][rdn_batch])
            b=np.array(B[aux][rdn_batch])
            print(np.shape(B))
            print(np.shape(K))
            print(np.shape(L1))
            print(um)
            s.exit()
        surf=ax1.scatter3D(k, b, L1,label='Latent Activation 1' )
        surf=ax2.scatter3D(k, b, L2,label='Latent Activation 2' )
        surf=ax3.scatter3D(k, b, L3,label='Latent Activation 3' )
plt.show()
#plt.legend()
#plt.pause(2)
#plt.close()
Latent_values_Scynet()
print('wtf')
