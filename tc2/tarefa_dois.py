import numpy as np
from otimo import PenalidadeExterior, LagrangeanoAumentado, GradienteConjugado, SecaoAurea

# Matriz de perdas (B)
B = np.array([
    [0.000049, 0.000014, 0.000015],
    [0.000014, 0.000045, 0.000016],
    [0.000015, 0.000016, 0.000039]
])

# Demanda
D = 850

# ----------- Funções do problema -----------

# Função objetivo: custo total
def custo_total(P):
    P1, P2, P3 = P
    return (
        0.15 * P1**2 + 38 * P1 + 756 +
        0.1 * P2**2 + 46 * P2 + 451 +
        0.25 * P3**2 + 40 * P3 + 1049
    )

# Perda de potência
def perda_potencia(P):
    return P @ B @ P  # Produto matricial: Pᵗ B P

# ----------- Restrições -----------

# Geração total - perdas ≥ demanda
def restricao_demanda(P): return np.sum(P) - D - perda_potencia(P)

# Limites inferiores
def r1(P): return P[0] - 150
def r2(P): return P[1] - 100
def r3(P): return P[2] - 50

# Limites superiores
def r4(P): return 600 - P[0]
def r5(P): return 400 - P[1]
def r6(P): return 200 - P[2]

# Todas as restrições
restricoes = [restricao_demanda, r1, r2, r3, r4, r5, r6]
# Penalidade Exterior usa todos como '>'
tipos_penalidade = np.array(['>', '>', '>', '>', '>', '>', '>'])
# Lagrangeano Aumentado trata demanda como '='
tipos_lagrangeano = np.array(['=', '>', '>', '>', '>', '>', '>'])

# Ponto inicial
x0 = np.array([300.0, 300.0, 150.0])

# Métodos de busca
busca_1d = SecaoAurea(precisao=1e-6)
irrestrito = GradienteConjugado(busca_1d, precisao=1e-6)

# ----------- Método 1: Penalidade Exterior -----------

def resolver_penalidade_exterior():
    metodo = PenalidadeExterior(precisao=1e-6)
    sol = metodo.resolva(
        custo_total, x0, restricoes, tipos_penalidade,
        irrestrito, penalidade=1.0, aceleracao=2.0
    )
    return sol

# ----------- Método 2: Lagrangeano Aumentado -----------

def resolver_lagrangeano_aumentado():
    metodo = LagrangeanoAumentado(precisao=1e-6)
    sol = metodo.resolva(
        custo_total, x0, restricoes, tipos_lagrangeano,
        irrestrito, penalidade=1.0, aceleracao=2.0
    )
    return sol

# ----------- Execução -----------

print("=== Método 1: Penalidade Exterior ===")
res1 = resolver_penalidade_exterior()
P1 = res1.x
print("P =", P1)
print("Custo total =", custo_total(P1))
print("Perda =", perda_potencia(P1))
print("Soma das potências =", np.sum(P1))
print("Demanda + perda =", D + perda_potencia(P1))

print("\n=== Método 2: Lagrangeano Aumentado ===")
res2 = resolver_lagrangeano_aumentado()
P2 = res2.x
print("P =", P2)
print("Custo total =", custo_total(P2))
print("Perda =", perda_potencia(P2))
print("Soma das potências =", np.sum(P2))
print("Demanda + perda =", D + perda_potencia(P2))
