import random as rd;import numpy as np;import sys as s;import pickle
import torch
import torch.nn as nn
from mpl_toolkits import mplot3d
from matplotlib import cm
import torch.optim as optim
import matplotlib.pyplot as plt
import math

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
#   GERANDO O BANCO DE DADOS SORTEANDO K E B NO SEU INTERVALO ESPECÍFICO

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
Predict_test_Scynet()
sys.exit()
