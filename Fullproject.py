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

def DampedPend(b,k,t,m):
    if t==0:
        position=1
    else:
        dump=math.exp(-(b/2*m)*t)
        omega=np.sqrt(k/m)*np.sqrt(1-(b**2)/(4*m*k))
        osc=np.cos(omega*t)
        position = dump*osc
    #   osc=1*np.cos((np.sqrt(k/m)*np.sqrt(1-(b**2)/(m*4*k))))*t)
    return position
def Box_1_dataset_with_Constants(n_batch,batch_size,exemplos_por_batch):
    inp=[];    question=[];    m=1
    T=[i for i in range(0,50)]
    K=np.linspace(5, 11, num=50)
    B=np.linspace(0.5, 1.1, num=50)
    KK=[];    BB=[]
#    K=np.linspace(5, 11, num=100)          #those are default values
#    B=np.linspace(0.5,1.1, num=100)        #those are default values
#'''         THIS IS FOR A RANDOM CONFIG OF K AND B'''
    for i in range(n_batch):
        t=[];        position=[];        full=0
        while full!=batch_size:
            ki=rd.randint(0,49);        bi=rd.randint(0,49)
            k=K[ki];        b=B[bi]
            KK.append(k);           BB.append(b)
            y=[];            tpred=[]
            for l in T:
                yy=DampedPend(b,k,l,m)
                y.append(yy)
                tpred.append(l)
#            plt.clf()                   #uncoment to graph
#            plt.xlim([0, 50])           #uncoment to graph
#            plt.ylim([-1, 1])           #uncoment to graph
#            plt.plot(tpred,y)           #uncoment to graph
#            plt.pause(0.5)              #uncoment to graph

            t.append(tpred)
            position.append(y)
            full+=1
        inp.append(position)
        question.append(t)
#        sys.exit()
    KK=np.array(KK).reshape(n_batch,batch_size,1)   # To works on scynet
    BB=np.array(BB).reshape(n_batch,batch_size,1)   # To works on scynet
    Constantes=[KK,BB]
#    print(np.shape(Constantes))
#    sys.exit()
    inp=torch.as_tensor(inp)
    question=torch.as_tensor(question)
#    plt.show()
    print('shape(question) =',np.shape(question))
    address = open("positions","wb")
    pickle.dump(inp, address)
    address.close()
    address = open("question","wb")
    pickle.dump(question, address)
    address.close()
    print('Constantes =',np.shape(Constantes))
    address = open("Constantes","wb")
    pickle.dump(Constantes, address)
    address.close()
Box_1_dataset_with_Constants(5,1000,50)
#s.exit()
#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
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
#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#inp=inp.reshape(2500,50)
#inp=inp.reshape(5,500,50)

#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#print(np.shape(inp))
#s.exit()
#-----------------------------------------------------------------------

#------------------ORGANIZE DATA-----------------------------------------------------
#-----------------------------------------------------------------------
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
            nn.Linear(200,3),#> latent
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
        #print(np.shape(t))
        #print(np.shape(encoded))
        aux=torch.cat((encoded,t),1)
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
    inp = pickle.load( open( "positions", "rb" ) )
    question= pickle.load( open( "question", "rb" ) )
    out =  pickle.load( open( "positions", "rb" ) )
    n_batch=np.shape(inp)[0]
    batch_size=np.shape(inp)[1]
    n_examples=np.shape(inp)[2]
    T=question[0,0]
    t=torch.as_tensor(np.zeros((batch_size,1)))
    answ=torch.as_tensor(np.zeros((batch_size,1)))
    indicedografico=0
    for epoch in range(epochs):
        for batch_idx in range(n_batch):
            inputs = inp[batch_idx]
            inputs=inputs.float()
            t=t.float()
            out=out.float()
            r=rd.randint(0,49)
            for i in range(batch_size):
                r=rd.randint(0,49)
                t[i][0]=question[batch_idx,i,r]
                answ[i][0]=inp[batch_idx,i,r]
            #recon = model(inputs,question[0][:][r])
            #loss=torch.mean((recon-out[batch_idx][:][r])**2)
            recon,latent = model(inputs,t)
            loss=torch.mean((recon-answ)**2)
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
#treine(10000)
#print('end')
#--------------------------------------------------------------------------
#---------------------TENTANDO SALVAR--------------------------------------
#--------------------------------------------------------------------------
PATH_save='Estado_talvez_funcionante5.pt'
PATH_load='Estado_talvez_funcionante4.pt'
#torch.save(model.state_dict(), PATH_save)

model.load_state_dict(torch.load(PATH_load))

#--------------------------------------------------------------------------
#---------------------GRÁFICOS---------------------------------------------
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
#s.exit()
#--------------------------------------------------------------------------
def Latent_values_Scynet():
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    t=torch.as_tensor(np.zeros((batch_size,1)))
    t=t.float()
    r=25 # tempo escolhido para a pergunta da rede neural
    for i in range(batch_size):
        t[i][0]=question[0,i,r]
    L1,L2,L3=np.zeros(batch_size),np.zeros(batch_size),np.zeros(batch_size)
    Y=np.zeros(50);        T=[i for i in range(0,50)]
    ks=[]
    bs=[]
    for aux in range(1,2):#n_batch):
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
