import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg
import sympy

# 计算两向量之差的模
def v_minus_norm(a,b):  
    if(isinstance(a[0],sympy.Symbol) or isinstance(a[0],sympy.Add) or isinstance(a[0],sympy.Pow) \
        or isinstance(b[0],sympy.Symbol) or isinstance(b[0],sympy.Add) or isinstance(b[0],sympy.Pow)):
        x, y = (a[0]-b[0], a[1]-b[1])
        return sympy.sqrt(x**2 + y**2)
    else:
        x, y = (abs(a[0]-b[0]) , abs(a[1]-b[1]))
        return math.sqrt(x**2 + y**2)

# 径向基函数
def h(x):
    if(x == 0):
        return 0
    if isinstance(x,sympy.Add) or isinstance(x,sympy.Symbol) or isinstance(x,sympy.Pow):
        return (x**2)*sympy.log(x)    
    else:
        return (x**2)*math.log(x)

# 求H矩阵
def get_H(P):
    k = len(P)
    H = []
    for i in range(k):
        t = []
        for j in range(k):
            if(i == j):
                t.append(0)
                continue
            tv = v_minus_norm(P[i],P[j])
            hij = h(tv)
            t.append(hij)
        H.append(t)
    return H

# 解矩阵方程 H*w=R 求得权值w
def get_w(H,R):
    w = linalg.solve(H, R)
    return w

# 求隐函数s
# 返回一个函数s
def get_s(w,P):
    """
    g是自变量 g = (x, y)
    """
    m = len(w)
    def s(x, y):
        res = 0
        g = (x, y)
        for i in range(m):
            t = v_minus_norm(g, P[i])
            res = res + w[i]*h(t)
        return res
    return s

# 测试数据
# P为约束点集
# P为每个约束点的类型：内点/边界点/外点

# 第一组
# P = [(0,0),(-2.5,-4),(-4,0),(-2,2),(0,4),(2,2),(4,0),(2,-2),(0,-4),(-3,3),(2,4),(2,-4)]
# R = [1,0,0,0,0,0,0,0,0,-1,-1,-1]

# 第二组
# P = [(17,4),(15,4),(19,4),(14,3),(15,3),(14,5),(15,5)]
# R = [1,0,0,-1,-1,-1,-1]
# 无法加padding

# 第三组 peanut
# P = [(0,0),(2,0),(-2,0),(0,-1),(0.6,-0.6),(2,-2),(3.4,-1.4),(4,0),(3,1.7),(2,2),(0,1),(-2,2),(-4,0),(-3.4,-1.4),(-2,-2),(-1,-1.7),(0,-2),(0,2)]#
# R = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1]#
# 注意H矩阵不能为奇异的

# 生成矩阵H
H = get_H(P)
# 求得权值集
w = get_w(H,R)
# print("w",w)

# 由此得到隐函数 s(x, y)
s = get_s(w,P)


# 生成画图
x, y = sympy.symbols('x y')
# padding = 5
x_var = (x,min(P,key=lambda x:x[0])[0], max(P,key=lambda x:x[0])[0] )
y_var = (y,min(P,key=lambda x:x[1])[1], max(P,key=lambda x:x[1])[1])
sympy.plot_implicit(s(x,y),x_var=x_var,y_var=y_var)




