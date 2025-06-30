"""
                            PEQUENO TESTE 06

Neste teste, vamos resolver um problema de otimização com restrições usando
o Método do Lagrangeano Aumentado. O problema é dado por:

        min f(x) = (x1^2)^(x2^2 + 1) + (x2^2)^(x1^2 + 1)
            s.a.
                g(x) = -(x1-1)^3 - x2 + 1 <= 0
                h(x) = (x1-2.2)^2 - x2 - 1 = 0

Neste teste, o seu objetivo é implementar a versão modificada da
função-objetivo na linha 57 e a atualização dos multiplicadores de Lagrange
nas linhas 72 e 73.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import optimize

# Definição da função-objetivo
def f(x):
    x1, x2 = x
    return (x1**2)**(x2**2 + 1) + (x2**2)**(x1**2 + 1)

# Definição da restrição de desigualdade
def g(x):
    x1, x2 = x
    return -(x1-1)**3 - x2 + 1

# Definição da restrição de igualdade
def h(x):
    x1, x2 = x
    return (x1-2.2)**2 - x2 - 1

# Parâmetros iniciais
x = np.array([3, 3], dtype=float) # Ponto inicial
mu = 0. # Multiplicador de Lagrange (desigualdade) p/ g1
lam = 0. # Multiplicador de Lagrange (igualdade) p/ h1
u = 1. # Constante de penalidade
alpha = 1.2 # Aceleração da penalização
k = 1 # Contador de iterações
precisao = 1e-2
historia = [x] # Histórico das variáveis de decisão
xanterior = x.copy() # Valor anterior da variável de decisão

# Processo iterativo
while True:
    
    # Definição da função Lagrangeana Aumentada que será a
    # função-objetivo do problema transformado
    def LA(x):
        fx = f(x)
        hx = h(x)
        gx = g(x)
        return ??????
    
    # Resolve problema de otimização irrestrito
    solution = optimize.minimize(LA, x, method='BFGS')
    
    # Atualização da variável de decisão
    x = solution.x
    
    print('Iteração %d' % k, end=' - ')
    print('x-ótimo: ' + str(x), end=', ')
    print('lambda = %.2f' % lam, end=', ')
    print('mu = %.2f' % mu, end=', ')
    print('u = %.2f' % u)
    
    # Atualização dos multiplicadores de Lagrange
    mu = ????
    lam = ????

    # Atualização da constante de penalização
    u = alpha*u

    # Atualização das iterações
    k += 1
    
    # Salva informação do novo ponto encontrado
    historia.append(x)
    
    # Verifica critério de parada
    if np.linalg.norm(x-xanterior)/np.linalg.norm(x) < precisao:
        break
    else:
        xanterior = x.copy()

"""           Visualização da trajetória do algoritmo                """

# Limites das variáveis de decisão
x1lim = (-1, 4)
x2lim = (-1, 4)
x1, x2 = np.meshgrid(np.linspace(*x1lim, 100),
                     np.linspace(*x2lim, 100))

# Calcula o valor da função-objetivo em cada ponto
fx = np.zeros_like(x1)
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        fx[i,j] = f([x1[i,j], x2[i,j]])

# Exibe as curvas de nível da função-objetivo
plt.figure()
levels = np.logspace(-1, 15, 20)  # Usando escala logarítmica para níveis
plt.contour(x1, x2, fx, levels=levels, norm=LogNorm(), cmap='hot')

# Exibe a restrição de desigualdade	
x1 = np.linspace(-1, 4, 100)
x2 = -(x1-1)**3 + 1
plt.plot(x1, x2, 'b-', label=r'$g(\mathbf{x})$')

# Exibe a restrição de igualdade
x2 = (x1-2.2)**2 - 1
plt.plot(x1, x2, 'g--', label=r'$h(\mathbf{x})$')

# Exibe a trajetória do algoritmo
history = np.array(historia)
plt.plot(history[:, 0], history[:, 1], '*--', color='dimgray')

# Marca a solução ótima do problema
plt.plot(x[0], x[1], '*k', label='Ótimo', markersize=15)
plt.legend()

# Configurações adicionais
plt.xlim(*x1lim)
plt.ylim(*x2lim)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Problema-Exemplo')
plt.legend()
plt.grid()
plt.show()