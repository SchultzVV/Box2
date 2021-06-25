import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn as nn
import pickle
import random as rd
import torch.optim as optim
import numpy as np;import sys as s
import matplotlib.pyplot as plt
import math
#           GERANDO O BANCO DE DADOS
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def Box_2_dataset_O(n_batch,batch_size,interval_size):
    T=[i for i in range(0,interval_size)]
    O=[];    Q=[];    A=[];    J=[]
    v_min,v_max=5,40
    velocidades=np.linspace(v_min,v_max,num=100)
    for i in range(n_batch*batch_size):
        v_rand1=rd.randint(len(velocidades)/2,len(velocidades)-1)
        v_rand2=rd.randint(0,len(velocidades)-1-len(velocidades)/2)
        v_free_i=velocidades[v_rand1]
        v_rot_i=velocidades[v_rand2]
        J1=velocidades[v_rand1]-velocidades[v_rand2]
        J.append(J1)
        #print('J1 = ',J1)
        random_list=np.linspace(0,J1,num=20)
        aux_rand=rd.randint(0,len(random_list)-1)
        v_free_f=random_list[aux_rand]
        v_rot_f=J1-v_free_f
        #print('v_free_f = ',v_free_f)
        #print('v_rot_f = ',v_rot_f)
        q_rot_i   = [velocidades[v_rand2]*(len(T)-i-1) for i in T]
        q_free_i  = [-velocidades[v_rand1]*(len(T)-i-1) for i in T]
        q_free_f  = [v_free_f*(i) for i in T]
        q_rot_f   = [v_rot_f*(i) for i in T]
        #print('v_free_i = ',v_free_i)
        #print('v_rot_i = ',v_rot_i)
        #print('J1 = ',J1)
        #print('v_free_f = ',v_free_f)
        #print('v_rot_f = ',v_rot_f)
        #print('q_free_i = ',q_free_i)
        #print('q_rot_i = ',q_rot_i)
        #print('q_free_f = ',q_free_f)
        #print('q_rot_f = ',q_rot_f)
        tpred=rd.randint(0,interval_size-1)
        q=[tpred]
        a=0
        for i in q_free_f:
            q.append(T[a])
            a+=1
            q.append(i)
        Q.append(q)
        a=[q_rot_f[tpred]]
        A.append(a)
        o=[];    a=0
        for i in q_rot_i:
            o.append(T[a])
            a+=1
            o.append(i)
        a=0
        for i in q_free_i:
            o.append(T[a])
            a+=1
            o.append(i)
        O.append(o)
    O=np.array(O).reshape(n_batch,batch_size,interval_size*4)
    Q=np.array(Q).reshape(n_batch,batch_size,interval_size*2+1)
    A=np.array(A).reshape(n_batch,batch_size,1)
    J=np.array(J).reshape(n_batch,batch_size,1)
    O=torch.as_tensor(O);    Q=torch.as_tensor(Q);    A=torch.as_tensor(A)
    print('np.shape(J)',np.shape(J));    print('np.shape(O)',np.shape(O))
    print('np.shape(Q)',np.shape(Q));    print('np.shape(A)',np.shape(A))
#    s.exit()
    address = open("O","wb");    pickle.dump(O, address);    address.close()
    address = open("Q","wb");    pickle.dump(Q, address);    address.close()
    address = open("A","wb");    pickle.dump(A, address);    address.close()
    address = open("J","wb");    pickle.dump(J, address);    address.close()

Box_2_dataset_O(5,500,5)
#s.exit()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#-------------------------------------------------------------------------------
#-----------------LOAD DATA-----------------------------------------------------
#-------------------------------------------------------------------------------
inp = pickle.load(open("O","rb"))
question= pickle.load(open("Q","rb"))
out =  pickle.load(open("A","rb"))
J = pickle.load(open("J","rb"))
n_batch=np.shape(inp)[0]
batch_size=np.shape(inp)[1]
n_examples=np.shape(inp)[2]
J=np.array(J).reshape(n_batch,batch_size,1)
Q_shape=np.shape(question)
Constantes =  pickle.load( open( "Constantes", "rb" ) )
K=Constantes[0];B=Constantes[1]
#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#inp=inp.reshape(2500,50)
#inp=inp.reshape(5,500,50)

#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#print(Q_shape[2])
#print(Q_shape[2]+1)
#s.exit()
#-------------------------------------------------------------------------------

#------------------ORGANIZE DATA------------------------------------------------
#-------------------------------------------------------------------------------
#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#s.exit()
#train_loader = torch.utils.data.DataLoader(inp, batch_size=batch_size)
#-------------------------------------------------------------------------------
#------------------DEFINE O MODELO----------------------------------------------
#-------------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        # N, 50
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_examples,400),
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
            nn.Linear(200,10),#> latent
            #nn.ELU(),
            #nn.ReLU(),
            nn.Tanh(),
            #nn.Linear(10,3),
            #nn.ELU(),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Linear(25,10),
        )
        self.project=nn.Linear(Q_shape[2],4)
        self.decoder=nn.Sequential(
            #nn.Tanh(),
            #nn.ReLU(),
            nn.Linear(14,100),
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
        #print(np.shape(t))
        #print(np.shape(encoded))
        aux=torch.cat((encoded,t),1)
        #print(np.shape(aux))
        decoded = self.decoder(aux)
        return decoded,encoded

#-------------------------------------------------------------------------------
#------------------UTILIZA O MODELO E INICIA CAMADAS DE PESOS ORTOGONAIS--------
#-------------------------------------------------------------------------------
model = Autoencoder()
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
criterion = nn.MSELoss() #segundo a investigar
optimizer = torch.optim.Adam(model.parameters())#,lr=1e-4,weight_decay = 1e-5)
#optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,weight_decay = 1e-5)#,momentum=0.5)
outputs = []
#---------------------------------------------------------------------------
#------TREINO DO DECODER MODIFICADO--INP[50]+TPRED >> OUT[1]----------------
#---------------------------------------------------------------------------
def treine(epochs):
    inp = pickle.load( open( "O", "rb" ) )
    question= pickle.load( open( "Q", "rb" ) )
    out =  pickle.load( open( "A", "rb" ) )
    n_batch=np.shape(inp)[0]
    batch_size=np.shape(inp)[1]
    n_examples=np.shape(inp)[2]
    Q_shape=np.shape(question)
    answ=torch.as_tensor(np.zeros((batch_size,1)))
    indicedografico=0
    for epoch in range(epochs):
        for batch_idx in range(n_batch):
            O=inp[batch_idx]
            Q=question[batch_idx]
            A=out[batch_idx]
            O=O.float()
            Q=Q.float()
            A=A.float()
            #recon = model(inputs,question[0][:][r])
            #loss=torch.mean((recon-out[batch_idx][:][r])**2)
            #print(np.shape(Q))
            #s.exit()
            recon,latent = model(O,Q)

            loss=torch.mean((recon-A)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #if indicedografico==3:
        #    Y=np.zeros(50);        T=[i for i in range(0,50)]
        #    rdn_batch=rd.randint(0,batch_size)
        #    YY=inp[0][rdn_batch].detach().numpy()
        #    r=0
        #    for interval in range(0,49):
        #        y=model(inp[0].float(),t).detach().numpy()[rdn_batch]
        #        Y[interval]=y
        #        r+=1
        #    plt.clf()
        #    plt.xlim([0, 15])
        #    plt.ylim([-1, 1])

        #    plt.scatter(T, Y,c='black',label='recon')
        #    plt.scatter(T, YY,c='red',label='answ')
        #    plt.legend()
        #    indicedografico=0
        #    plt.pause(0.03)
        #    #plt.close()
        #indicedografico+=1
        print(f'Epoch:{epoch+1},Loss:{loss.item():.4f}')
        #outputs.append((epoch,inputs,recon))

#    plt.show()
#treine(1000)
#print('end')
#--------------------------------------------------------------------------
#---------------------TENTANDO SALVAR--------------------------------------
#--------------------------------------------------------------------------
PATH_save='Estado_Box_2.pt'
PATH_load='Estado_Box_2.pt'
model.load_state_dict(torch.load(PATH_load))
treine(300)
torch.save(model.state_dict(), PATH_save)
s.exit()
#--------------------------------------------------------------------------
#---------------------GRÁFICOS---------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
def Latent_values_Scynet():
    for aux in range(1,n_batch):
        O=inp[aux].float()
        Q=question[aux].float()
        A=out[aux].float()
        for rdn_batch in range(0,batch_size):
            recon,latent=model(inp[aux].float(),t)
            L1[rdn_batch] = latent[rdn_batch][0].detach().numpy()
            L2[rdn_batch] = latent[rdn_batch][1].detach().numpy()
            L3[rdn_batch] = latent[rdn_batch][2].detach().numpy()
            ks.append(K[aux][rdn_batch])
            bs.append(B[aux][rdn_batch])
        surf=ax1.scatter3D(ks, bs, L1,label='Latent Activation 1')
        surf=ax2.scatter3D(ks, bs, L2,label='Latent Activation 2')
        surf=ax3.scatter3D(ks, bs, L3,label='Latent Activation 3')
        ks=[]
        bs=[]
#        plt.pause(5)
    plt.show()
#Latent_values_Scynet()
#s.exit()
#--------------------------------------------------------------------------
def Latent_values_Scynet_html_graph():
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
#plt.pause(10)
        #plt.close()
    #plt.legend()
Latent_values_Scynet_html_graph()
s.exit()
print('wtf')
