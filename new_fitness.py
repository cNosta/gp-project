import numpy as np
import matplotlib.pyplot as plt
import math
import random

dmin = 1
dmax = 2
Np = 20
Nt = 1
No = 2

# for testing

target = [(0,0)] # 设置target坐标
obstacle = [] # 设置窄道的位置(窄道的)

def get_pattern():
    def pattern(x,y):
        return np.exp(-(x-2)**2-(y-2)**2) + 1.2*np.exp(-x**2-y**2)
    return pattern

def get_k():
    return random.randint()
# end

# formal
def get_A(dpt):
    A = [[0 for i in range(Np)] for j in range(Nt)]
    for i in range(Np):
        for j in range(Nt):
            if(dpt[i][j] <= dmax and dpt[i][j] >= dmin):
                A[i][j] = 1
    return A

def get_pos(pattern):
    # 1.由pattern生成等高线
    # 2.在等高线上采样生成机器人坐标pos
   
    x = np.arange(-1.5,3.5,0.1)
    y = np.arange(-1.5,3.5,0.1)
    X, Y = np.meshgrid(x,y)
    Z = pattern(X, Y)
    N = np.arange(-0.2, 1.5, 0.1)
    CS = plt.contour( Z, N, linewidth=2, cmap = mpl.cm.jet )
    # plt.show()
    path = CS.collections[10].get_paths()[10]
    vertices = path.vertices
    px = vertices[:,0]
    py = vertices[:,1]
    p_pos = []
    for i, j in px, py:
        p_pos.append(zip(i, j))
    return p_pos

# 计算每个机器人到target的距离
def get_dpt(pos):
    dpt = [ math.sqrt((tx-px)**2+(ty-py)**2)  for tx, ty in target for px, py in pos]
    return dpt

# 计算每个机器人到obstacle的距离
def get_dpo(pos):
    dpo = [ math.sqrt((ox-px)**2+(oy-py)**2)  for ox, oy in obstacle for px, py in pos]
    return dpo

def get_domin(dpo):
    return min( map(min,dpo) )

def sigmoid(x,a,k):
    return 1/(1+np.exp(-k*(x-a)))

def fitness(k, pattern):
    """
    GP优化而来的k值和pattern式子
    """
    
    dpt, pos = get_dpt(pattern) # 列表dpt[i][j] 表示第i个机器人到第j个目标的距离
    dpo = get_dpo(pos) # 列表dpo[i][j] 表示第i个机器人到第j个障碍物的距离
    domin = get_domin(dpo) # 机器人到障碍物的最小距离
    A = get_A(dpt)
    t1 = 0
    t2 = 0
    for i in range(Np):
        for j in range(Nt):
            t1 = t1 + ( sigmoid(dpt[i][j],dmax,k[1])+sigmoid(dmin,dpt[i][j],k[2])-A[i][j] )/( Np*Nt*1.0 )
    for i in range(Np):
        for j in range(No):
            t2 = t2 + sigmoid(domin, dpo[i][j],k[3])/(Np*N0*1.0)
    f = t1 + t2
    return f


pattern = get_pattern()
k = get_k()
