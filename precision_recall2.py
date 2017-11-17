import numpy as np
import matplotlib.pyplot as plt

#just a dummy sample
recall=np.linspace(0.0,1.0,num=42)
precision=np.random.rand(42)*(1.-recall)
precision2=precision.copy()
i=recall.shape[0]-2

# interpolation...
while i>=0:
    if precision[i+1]>precision[i]:
        precision[i]=precision[i+1]
    i=i-1

# plotting...
fig, ax = plt.subplots()
for i in range(recall.shape[0]-1):
    ax.plot((recall[i],recall[i]),(precision[i],precision[i+1]),'k-',label='',color='red') #vertical
    ax.plot((recall[i],recall[i+1]),(precision[i+1],precision[i+1]),'k-',label='',color='red') #horizontal

ax.plot(recall,precision2,'k--',color='blue')
#ax.legend()
ax.set_xlabel("recall")
ax.set_ylabel("precision")
plt.savefig('fig.jpg')
fig.show()
input()