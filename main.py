import numpy as np
import matplotlib.pyplot as plt

# Parametrii problemei
DIMENSIUNE_POPULATIE = 100
GENERATII = 50
RATA_INCRUCISARE = 0.9
RATA_MUTATIE = 0.1


# Intervalele variabilelor de decizie
LIMITA_PUTERE = [50, 300]  # Puterea motorului (kW)
LIMITA_AERODINAMICA = [0.2, 0.8]  # Coeficientul aerodinamic
LIMITA_GREUTATE = [800, 2000]  # Greutatea vehiculului (kg)


# Functiile obiectiv
def functii_obiectiv(individ):
    putere, coeficient, greutate = individ
    consum_combustibil = (greutate * coeficient) / putere
    viteza_maxima = putere / (coeficient * greutate)
    return consum_combustibil, viteza_maxima



# Initializarea populatiei
def initializare_populatie(dimensiune):
    populatie = []
    for _ in range(dimensiune):
        putere = np.random.uniform(*LIMITA_PUTERE)
        coeficient = np.random.uniform(*LIMITA_AERODINAMICA)
        greutate = np.random.uniform(*LIMITA_GREUTATE)
        populatie.append([putere, coeficient, greutate])
    return np.array(populatie)

# Sortarea nedominata
def sortare_nedominata(populatie):
    fronturi = []
    numar_dominatii = np.zeros(len(populatie))
    solutii_dominante = [[] for _ in range(len(populatie))]

    for i in range(len(populatie)):
        for j in range(len(populatie)):
            f1_i, f2_i = functii_obiectiv(populatie[i])
            f1_j, f2_j = functii_obiectiv(populatie[j])

            if (f1_i < f1_j and f2_i > f2_j) or (f1_i <= f1_j and f2_i > f2_j) or (f1_i < f1_j and f2_i >= f2_j):
                solutii_dominante[i].append(j)
            elif (f1_j < f1_i and f2_j > f2_i) or (f1_j <= f1_i and f2_j > f2_i) or (f1_j < f1_i and f2_j >= f2_i):
                numar_dominatii[i] += 1

        if numar_dominatii[i] == 0:
            fronturi.append([i])

    return fronturi



# Calcularea distantei de aglomerare
def distanta_aglomerare(front, populatie):
    distante = np.zeros(len(front))
    obiective = np.array([functii_obiectiv(populatie[i]) for i in front])

    for m in range(obiective.shape[1]):
        indices_sortate = np.argsort(obiective[:, m])
        distante[indices_sortate[0]] = distante[indices_sortate[-1]] = float('inf')
        for i in range(1, len(front) - 1):
            distante[indices_sortate[i]] += (
                obiective[indices_sortate[i + 1], m] - obiective[indices_sortate[i - 1], m]
            )

    return distante



# Operatori pentru selectie, incrucisare si mutatie
def incrucisare(parinte1, parinte2):
    if np.random.rand() < RATA_INCRUCISARE:
        return (np.array(parinte1) + np.array(parinte2)) / 2
    return parinte1



def mutatie(individ):
    if np.random.rand() < RATA_MUTATIE:
        individ[0] += np.random.uniform(-10, 10)
        individ[1] += np.random.uniform(-0.05, 0.05)
        individ[2] += np.random.uniform(-50, 50)

        individ[0] = np.clip(individ[0], *LIMITA_PUTERE)
        individ[1] = np.clip(individ[1], *LIMITA_AERODINAMICA)
        individ[2] = np.clip(individ[2], *LIMITA_GREUTATE)

    return individ


populatie = initializare_populatie(DIMENSIUNE_POPULATIE)

for generatie in range(GENERATII):
    urmasi = []

    if len(populatie) < DIMENSIUNE_POPULATIE:
        populatie_suplimentara = initializare_populatie(DIMENSIUNE_POPULATIE - len(populatie))
        populatie = np.concatenate((populatie, populatie_suplimentara))

    for _ in range(DIMENSIUNE_POPULATIE // 2):
        parinte1, parinte2 = np.random.choice(range(len(populatie)), 2, replace=False)
        copil1 = incrucisare(populatie[parinte1], populatie[parinte2])
        copil2 = incrucisare(populatie[parinte2], populatie[parinte1])
        urmasi.append(mutatie(copil1))
        urmasi.append(mutatie(copil2))

    populatie = np.concatenate((populatie, urmasi))

    fronturi = sortare_nedominata(populatie)
    populatie_noua = []

    for front in fronturi:
        if len(populatie_noua) + len(front) > DIMENSIUNE_POPULATIE:
            distante = distanta_aglomerare(front, populatie)
            indici_sortati = np.argsort(-distante)
            populatie_noua.extend(
                [populatie[i] for i in indici_sortati[: DIMENSIUNE_POPULATIE - len(populatie_noua)]]
            )
            break
        populatie_noua.extend([populatie[i] for i in front])

    if len(populatie_noua) < DIMENSIUNE_POPULATIE:
        populatie_suplimentara = initializare_populatie(DIMENSIUNE_POPULATIE - len(populatie_noua))
        populatie_noua.extend(populatie_suplimentara)

    populatie = np.array(populatie_noua)

# Plotarea solutiilor
valori_f1, valori_f2 = zip(*[functii_obiectiv(individ) for individ in populatie])
plt.scatter(valori_f1, valori_f2, color="blue", label="Solutii Pareto")
plt.xlabel("Consumul de combustibil (f1)")
plt.ylabel("Viteza maxima (f2)")
plt.title("Frontul Pareto pentru optimizarea vehiculului")
plt.legend()
plt.show()