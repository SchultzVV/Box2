import torch
import timeit
t0= timeit.timeit()
import torch.nn as nn
import pickle
import random as rd
import torch.optim as optim
import numpy as np;import sys as s
from torchvision import datasets, transforms
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
def Box_1_dataset(n_batch,batch_size,exemplos_por_batch):
    inp=[];    question=[];    m=1
    T=[i for i in range(0,50)]
    K=np.linspace(5, 11, num=100)
    B=np.linspace(0.5, 1.1, num=100)
    for i in range(n_batch):
        t=[];        position=[];        full=0
        while full!=batch_size:
            ki=rd.randint(0,99);        bi=rd.randint(0,99)
            k=K[ki];        b=B[bi]
            y=[];            tpred=[]
            for l in T:
                yy=DampedPend(b,k,l,m)
                y.append(yy)
                tpred.append(l)
            #plt.clf()
            #plt.xlim([0, 50])
            #plt.ylim([-1, 1])
            #plt.plot(tpred,y)
            #plt.pause(0.0000001)

            #plt.close()
            t.append(tpred)
            position.append(y)
        #    print(np.shape(position))
            full+=1
        inp.append(position)
        print('shape(inp) =',np.shape(inp))
        question.append(t)
#        print(np.shape(inp))
#        print(np.shape(question))
        #sys.exit()
    inp=torch.as_tensor(inp)
    question=torch.as_tensor(question)
    #plt.show()
    print('shape(question) =',np.shape(question))
    address = open("positions","wb")
    pickle.dump(inp, address)
    address.close()
    address = open("question","wb")
    pickle.dump(question, address)
    address.close()
#Box_1_dataset(5,500,50)
#s.exit()
#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#-----------------------------------------------------------------------
#------------------LOAD DATA-----------------------------------------------------
#-----------------------------------------------------------------------
inp = pickle.load( open( "positions", "rb" ) )
question= pickle.load( open( "question", "rb" ) )
out =  pickle.load( open( "positions", "rb" ) )
#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#inp=inp.reshape(2500,50)
#inp=inp.reshape(5,500,50)

#plt.plot(question[1][0].detach().numpy(),out[2][0].detach().numpy())
#plt.show()
#print(np.shape(inp))
#s.exit()
n_batch=np.shape(inp)[0]
batch_size=np.shape(inp)[1]
n_examples=np.shape(inp)[2]
t=question[0][0]
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
        return decoded

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
            recon = model(inputs,t)
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

    plt.show()
#treine(10000)
#print('end')
#--------------------------------------------------------------------------
#---------------------TENTANDO SALVAR--------------------------------------
#--------------------------------------------------------------------------
PATH_save='Estado_talvez_funcionante5.pt'
PATH_load='Estado_talvez_funcionante4.pt'
#torch.save(model.state_dict(), PATH_save)
#print('loss=',loss)
#tf= timeit.timeit()

#print('o tempo que levou =',tf-t0)
s.exit()
#s.exit()
model.load_state_dict(torch.load(PATH_load))
#state = {
#    'epoch': epoch,
#    'state_dict': model.state_dict(),
#    'optimizer': optimizer.state_dict(),
#    ...
#}
#torch.save(state, filepath)
#--------------------------------------------------------------------------
#---------------------GRÁFICOS---------------------------------------------
#--------------------------------------------------------------------------
def testes():
    Y=np.zeros(50);        T=[i for i in range(0,50)]
    t=torch.as_tensor(np.zeros((batch_size,1)))
    t=t.float()
    answ=torch.as_tensor(np.zeros((batch_size,1)))
    for _ in range(0,n_batch*batch_size):
        rdn_n_batch=rd.randint(0,n_batch-1)
        rdn_batch=rd.randint(0,batch_size-1)
        YY=inp[rdn_n_batch][rdn_batch]#.detach().numpy()
        r=0
        for interval in range(0,49):
            for i in range(batch_size):
                t[i][0]=question[0,i,interval]
            y=model(inp[rdn_n_batch].float(),t)[rdn_batch]      #.detach().numpy()
            r+=1
            print(y)
            #print(y)
            Y[interval]=y
        #print(np.shape(t))
        #print(np.shape(Y))
    #    s.exit()
        #print(np.shape(YY))
        #print(np.shape(Y))
            plt.clf()
            plt.xlim([0, 50])
            plt.ylim([-1, 1])
            plt.scatter(T, Y,c='black',label='recon')
            plt.scatter(T, YY,c='red',label='answ')
            plt.plot(T, Y,'black')
            plt.plot(T, YY,'red')
            plt.pause(0.0000001)
            plt.legend()
        #plt.close()
    plt.show()
#testes()


print('wtf')
