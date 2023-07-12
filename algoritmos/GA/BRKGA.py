"""Genetic Algorithm."""
import copy
import random
from builtins import list
from contextlib import contextmanager
from datetime import datetime, timedelta
from math import ceil, fabs
import openpyxl

import matplotlib.pyplot as plt
import numpy
from deap import base
from deap import creator
from deap import tools

dadosExel = []
listImpar = []
listPar = []
tempoExecution = 0
@contextmanager
def timeit(file_write=None):
    """Context Manager to check runtime."""
    start_time = datetime.now()
    # print(f'Tempo de Inicio (hh:mm:ss.ms) {start_time}', file=file_write)
    yield
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    # print(f'Tempo de Termino (hh:mm:ss.ms) {end_time}', file=file_write)
    print(f'Tempo Total (hh:mm:ss.ms) {time_elapsed}', file=file_write)
    tempoExecution = copy.deepcopy(time_elapsed)

def dist2pt(x1, y1, x2, y2):
    """."""
    return max(fabs(x2 - x1), fabs(y2 - y1))  # Distancia de Chebyschev
    # return ((x1 - x2)**2 + (y1 - y2)**2)**(1 / 2) # Distancia Euclidiana


def midPoint(x1, y1, x2, y2):
    """."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def plotar(indiv, f):
    """."""
    individuo = decode(indiv)
    # individuo = indiv
    fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    x1, y1, x, y = [], [], [], []
    colors = ['red', 'gray', 'yellow']
    cutA = 1
    i3 = individuo[0][0]
    a3 = edges[i3]
    if not individuo[1][0] == 0:
        a3 = edges[i3][::-1]
    if 'D' in a3:
        a3.remove('D')
        a3.append('D')
    deslocamentos = []
    x.append(a3[0][0])
    y.append(a3[0][1])
    x.append(a3[1][0])
    y.append(a3[1][1])

    if 'D' in a3:
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[2])
        f1_axes.annotate(str(cutA), midPoint(*a3[0], *a3[1]))
    else:
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[0])
        f1_axes.annotate(str(cutA), midPoint(*a3[0], *a3[1]))

    cutA += 1
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]  # aresta atual
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]  # proxima aresta
        a1 = edges[i1]
        if not individuo[1][i] == 0:
            a1 = edges[i1][::-1]  # atual

        if 'D' in a1:
            a1.remove('D')
            a1.append('D')

        a2 = edges[i2]
        if not individuo[1][i + 1 if i + 1 < len(individuo[0]) else 0] == 0:
            a2 = edges[i2][::-1]  # proxima
        if 'D' in a2:
            a2.remove('D')
            a2.append('D')

        x1, y1, x, y = [], [], [], []
        isDeslocamento = a1[1] != a2[0]
        if isDeslocamento:  # se a proxima não comecar onde a primeira termina
            x1.append(a1[1][0])
            y1.append(a1[1][1])
            x1.append(a2[0][0])
            y1.append(a2[0][1])
            # print(edges[i1][1], edges[i2][0], i1, i2)
            # plt.annotate("Des-"+str(deslocamento), midPoint(
            #     *edges[i1][1], *edges[i2][0]))
            # deslocamento += 1
            # plt.plot(x1, y1, '-*', color=colors[1])
            # f1_axes.quiver(x1[0], y1[0], x1[1] - x1[0], y1[1] - y1[0], linestyle=':',
            #                scale_units='xy', angles='xy', scale=1, color=colors[1])
            # f1_axes.annotate(str(cutA), midPoint(*a1[1], *a2[0]))
            deslocamentos.append({
                'pontos': [x1[0], y1[0], x1[1] - x1[0], y1[1] - y1[0]],
                'annot': str(cutA),
                'mid': midPoint(*a1[1], *a2[0])
            })
            cutA += 1
        # plota a proxima
        x.append(a2[0][0])
        y.append(a2[0][1])
        x.append(a2[1][0])
        y.append(a2[1][1])
        # plt.annotate(str(cutA), midPoint(
        #     *a2[0], *a2[1]))
        # plt.plot(x, y, '-*', color=colors[0])
        if 'D' in a2:
            f1_axes.annotate(str(cutA), midPoint(*a2[0], *a2[1]))
            f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[2])
        else:
            f1_axes.annotate(str(cutA), midPoint(*a2[0], *a2[1]))
            f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0],
                       scale_units='xy', angles='xy', scale=1, color=colors[0])
        cutA += 1
    for i in deslocamentos:
        f1_axes.annotate(i['annot'], (i['mid'][0] - 3, i['mid'][1]))
        f1_axes.quiver(*i['pontos'], width=.005,
                       scale_units='xy', angles='xy', scale=1, color=colors[1])
    f1_axes.set_xlim(*f1_axes.get_xlim())
    f1_axes.set_ylim(*f1_axes.get_ylim())
    fig1.savefig(f'resultados/rco3/rco310/{f}.png', dpi=300)
    plt.close()

def selectNext(x, listKK):
    select = None
    for i in listKK:
        if i[0] == x:
            index = random.randint(1, len(i[1]))
            select = i[1].pop(index - 1)
            # while select[0] in listImpar and select[1] in listImpar and i[1] != []:
            #     i[1].append(select)
            #     index = random.randint(1, len(i[1]))
            #     select = None
            #     select = i[1].pop(index - 1)

            if len(i[1]) == 0:
                listKK.remove(i)
            break

    if select == None:
        index = random.randint(1, len(listKK))
        removedNone = listKK[index - 1]
        listRemovedNone = removedNone[1]
        index = random.randint(1, len(listRemovedNone))
        select = listRemovedNone.pop(index - 1)
        x = removedNone[0]
        if listRemovedNone == []:
            listKK.remove(removedNone)


    second = None


    if x == select[0]:
        second = select[1]
    else:
        second = select[0]

    for k in listKK:
        if k[0] == second:
            k[1].remove(select)
            if len(k[1]) == 0:
                listKK.remove(k)
            break

    return select

def direction(p1, p2):
    xMax = p1[0] > p2[0]
    xMin = p1[0] < p2[0]
    xEqual = p1[0] == p2[0]
    yMax = p1[1] > p2[1]
    yMin = p1[1] < p2[1]
    yEqual = p1[1] == p2[1]

    return xMax, xMin, xEqual, yMax, yMin, yEqual

def newDirectionNext(after, next, isFirst):
    if isFirst:
        if after[0] == next[0] or after[0] == next[1]:
            return 0.75
        if after[1] == next[0] or after[1] == next[1]:
            return 0.25
    else:
        if after[1] == next[0]:
            return 0.25
        if after[1] == next[1]:
            return 0.75
        return random.random()

def genIndividuoRK(edges):
    """
    Generate Individuo.

    args:
        edges -> edges to cut of grapth

    individuo[0]: order of edges
    individuo[1]: order of cut

    """
    individuos = []

    size = len(listImpar)
    if size != 2:
        individuos = [random.random() for i in range(len(edges))], [
            random.random() for i in range(len(edges))]
        return individuos

    listCopy = copy.deepcopy(edges)
    listj = listImpar
    listk = listPar

    listPoints = []
    for i in listj:
        listResult = []
        listEdges = []
        listResult.append(i)
        for j in listCopy:
            if i in j:
                if 'D' in j:
                    j.remove('D')
                listEdges.append(j)
        listResult.append(listEdges)
        listPoints.append(listResult)

    for i in listk:
        listResult = []
        listEdges = []
        listResult.append(i)
        for j in listCopy:
            if i in j:
                if 'D' in j:
                    j.remove('D')
                listEdges.append(j)
        listResult.append(listEdges)
        listPoints.append(listResult)

    index = random.randint(1, len(listj))
    select = listj[index -1]
    resultSelect = []
    while listCopy != []:

        result = selectNext(select, listPoints)
        if select == result[0]:
            select = result[1]
        else:
            select = result[0]
        resultSelect.append(result)
        listCopy.remove(result)

    p = []
    for _ in range(len(resultSelect)):
        numero = random.random()
        p.append(numero)
    p.sort(reverse=True)

    listIndividuos = [0.0] * len(resultSelect)

    for edgekk in resultSelect:
        if edgekk in edges:
            indexEdge = edges.index(edgekk)
            listIndividuos[indexEdge] = p.pop(0)
        else:
            edgekk.append('D')
            indexEdge = edges.index(edgekk)
            listIndividuos[indexEdge] = p.pop(0)

    listDirection = [0.0] * len(resultSelect)
    listResultDirection = []

    for i in range(len(resultSelect)):
        edgekk = resultSelect[i]
        if i == 0:
            result = newDirectionNext(edgekk, resultSelect[i + 1], True)
            if result < 0.5:
               listResultDirection.append(edgekk)
            else:
                listResultDirection.append([edgekk[1], edgekk[0]])
            listDirection[edges.index(edgekk)] = result
        else:
            nextIndex = i - 1
            nextEdgekk = listResultDirection[nextIndex]
            if edgekk in edges:
                result = newDirectionNext(nextEdgekk, edgekk, False)
                if result < 0.5:
                    listResultDirection.append(edgekk)
                else:
                    listResultDirection.append([edgekk[1], edgekk[0]])
                listDirection[edges.index(edgekk)] = result
            else:
                edgekk.append('D')
                result = newDirectionNext(nextEdgekk, edgekk, False)
                if result < 0.5:
                    listResultDirection.append(edgekk)
                else:
                    listResultDirection.append([edgekk[1], edgekk[0]])
                listDirection[edges.index(edgekk)] = result
    individuos.append(listIndividuos)
    individuos.append(listDirection)
    return individuos


def decode(ind):
    """."""
    listIndOrd = []
    listDirOrd = []
    indCopy = copy.deepcopy(ind[0])

    while indCopy != []:
        indexInd = 0
        max = indCopy[0]
        for i in range(len(indCopy)):
            if max < indCopy[i]:
                indexInd = i
                max = indCopy[indexInd]

        resultInd = indCopy.pop(indexInd)
        index = ind[0].index(resultInd)
        listIndOrd.append(index)
        listDirOrd.append(0 if ind[1][index] < 0.5 else 1)

    return[listIndOrd, listDirOrd]


def evalCut(individuo, pi=100 / 6, mi=400):
    """
    Eval Edges Cut.

    args:
        pi -> cutting speed
        mi -> travel speed

    if individuo[1][i] == 0 the cut is in edge order
    else the cut is in reverse edge order

    """
    ind = decode(individuo)

    dist = 0
    i3 = ind[0][0]

    a3 = edges[i3]
    if not ind[1][0] == 0:
        a3 = edges[i3][::-1]

    if 'D' in a3:
        a3.remove('D')
        a3.append('D')

    if a3[0] != (0.0, 0.0):
        dist += dist2pt(0.0, 0.0, *a3[0]) / mi

    if len(a3) == 3:
        dist += dist2pt(*a3[0], *a3[1]) / mi
    else:
        dist += (dist2pt(*a3[0], *a3[1])) / pi

    for i in range(len(ind[0]) - 1):
        i1 = ind[0][i]
        # i2 = ind[0][i + 1 if i + 1 < len(ind[0]) else 0]
        i2 = ind[0][i + 1]

        a1 = edges[i1]
        if not ind[1][i] == 0:
            a1 = edges[i1][::-1]
            if 'D' in a1:
                a1.remove('D')
                a1.append('D')

        a2 = edges[i2]

        if not ind[1][i + 1] == 0:
            a2 = edges[i2][::-1]
            if 'D' in a2:
                a2.remove('D')
                a2.append('D')
        if a1[1] == a2[0]:
            dist += (dist2pt(*a2[0], *a2[1])) / pi
        elif 'D' in a2:
            dist += (dist2pt(*a2[0], *a2[1])) / mi
        else:
            dist += (dist2pt(*a1[1], *a2[0])) / mi + (
                dist2pt(*a2[0], *a2[1])) / pi

    iu = ind[0][-1]
    au = edges[iu]
    if not ind[1][-1] == 0:
      au = edges[iu][::-1]

    if 'D' in au:
        au.remove('D')
        au.append('D')

    if au != (0.0, 0.0):
        dist += dist2pt(*au[1], 0.0, 0.0) / mi
    individuo.fitness.values = (dist, )
    return dist,


def main(P=1000, Pe=0.2, Pm=0.3, pe=0.7, NumGenWithoutConverge=100, file=None):
    """
    Execute Genetic Algorithm.

    args:
        P -> size of population
        Pe -> size of elite population
        Pm -> size of mutant population
        Pe -> elite allele inheritance probability
        NumGenWithoutConverge -> Number of generations without converge
        file -> if write results in file

    """
    tempo = timedelta(seconds=300)

    pop = toolbox.population(n=P)

    toolbox.register("mate", crossBRKGA, indpb=pe)

    tamElite = ceil(P * Pe)
    tamMutant = ceil(P * Pm)
    tamCrossover = P - tamElite - tamMutant

    gen, genMelhor = 0, 0

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Evaluate the entire population
    list(toolbox.map(toolbox.evaluate, pop))
    melhor = numpy.min([i.fitness.values for i in pop])
    logbook = tools.Logbook()
    p = stats.compile(pop)
    logbook.record(gen=0, **p)
    logbook.header = "gen", 'min', 'max', "avg", "std"
    gens, inds = [], []
    gens.append(gen)
    inds.append(melhor)
    # print(logbook.stream, file=file)
    hora = datetime.now()
    while gen - genMelhor <= NumGenWithoutConverge:
        # Select the next generation individuals
        offspring = sorted(
            list(toolbox.map(toolbox.clone, pop)),
            key=lambda x: x.fitness,
            reverse=True
        )
        elite = offspring[:tamElite]
        cross = offspring[tamElite:tamCrossover]
        c = []
        # Apply crossover and mutation on the offspring
        for _ in range(tamCrossover):
            e1 = random.choice(elite)
            c1 = random.choice(cross)
            ni = creator.Individual([[], []])
            ni[0] = toolbox.mate(e1[0], c1[0])
            ni[1] = toolbox.mate(e1[1], c1[1])
            c.append(ni)

        p = toolbox.population(n=tamMutant)
        c = elite + c + p
        offspring = c

        # Evaluate the individuals with an invalid fitness

        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # list(toolbox.map(toolbox.evaluate, invalid_ind))

        list(toolbox.map(toolbox.evaluate, offspring[tamElite:]))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        gen += 1
        minf = numpy.min([i.fitness.values for i in pop])
        men = False
        try:
            if minf < melhor:
                men = True
                melhor = minf
                genMelhor = gen
        except Exception:
            print(minf)

        p = stats.compile(pop)
        logbook.record(gen=gen, **p)
        # if gen - genMelhor <= NumGenWithoutConverge and not men:
        #     print(logbook.stream)
        # else:
        #     print(logbook.stream, file=file)
        hof.update(pop)
        gens.append(gen)
        inds.append(minf)

        if (datetime.now() - hora) > tempo:
            break

    return pop, stats, hof, gens, inds


def crossBRKGA(ind1, ind2, indpb):
    """."""
    return [ind1[i] if random.random() < indpb else ind2[i]
            for i in range(min(len(ind1), len(ind2)))]



def verifyEdges(vetice, listVertices, listEdger):

    x = None
    y = None

    res = None

    quantifyMax = 100
    quantify = 0

    for index in range(len(listVertices) - 1):
        vertex = None
        if vetice[0] != listVertices[index][0] and vetice[1] != listVertices[index][1]:
            vertex = [(vetice[0], vetice[1]), (listVertices[index][0], listVertices[index][1])]
            if vertex not in listEdger:
                result = dist2pt(vetice[0], listVertices[index][0], vetice[1], listVertices[index][1])
                if res is None:
                    res = result
                    x = listVertices[index][0]
                    y = listVertices[index][1]
                else:
                    if result < res:
                        quantify = 0
                        x = listVertices[index][0]
                        y = listVertices[index][1]
                    else:
                        quantify = quantify + 1
                if quantify > quantifyMax:
                    break

    return x, y

def removeOddVerteces(vertex, listImpar, listPar, listEdger):
    x = None
    y = None
    isOdd = True

    x, y = verifyEdges(vertex, listImpar, listEdger)

    if x is None and y is None:
        x, y = verifyEdges(vertex, listPar, listEdger)
        isOdd = False

    return x, y, isOdd

def preProcess(file):
    listEdger = []

    if file:
        length = int(file[0])
        for line in range(length):
            indexLine = line + 1
            vertices = [float(vertex) for vertex in file[indexLine].split()]
            if (vertices[0], vertices[1]) in listImpar and (vertices[0], vertices[1]) not in listPar:
                listImpar.remove((vertices[0], vertices[1]))
                listPar.append((vertices[0], vertices[1]))
            elif (vertices[0], vertices[1]) in listPar and (vertices[0], vertices[1]) not in listImpar:
                listPar.remove((vertices[0], vertices[1]))
                listImpar.append((vertices[0], vertices[1]))
            else:
                listImpar.append((vertices[0], vertices[1]))

            if(vertices[2], vertices[3]) in listImpar and (vertices[2], vertices[3]) not in listPar:
                listImpar.remove((vertices[2], vertices[3]))
                listPar.append((vertices[2], vertices[3]))
            elif(vertices[2], vertices[3]) in listPar and (vertices[2], vertices[3]) not in listImpar:
                listPar.remove((vertices[2], vertices[3]))
                listImpar.append((vertices[2], vertices[3]))
            else:
                listImpar.append((vertices[2], vertices[3]))

            listEdger.append([(vertices[0], vertices[1]), (vertices[2], vertices[3])])

    isPreProcess = random.random()
    if isPreProcess < 0.5:
        return listEdger

    while listImpar:
        if len(listImpar) == 2 or len(listImpar) == 0:
            break
        if len(listImpar) == 1:
            vertex = listImpar.pop(0)
            listEdger.append([vertex, vertex, 'D'])

        else:
            index = random.randint(0, len(listImpar) - 1)
            vertex = listImpar.pop(index)
            x, y, isOdd = removeOddVerteces(vertex, listImpar, listPar, listEdger)
            listPar.append(vertex)

            if isOdd:
                listImpar.remove((x, y))
                listPar.append((x, y))
            else:
                listImpar.append((x, y))
                listPar.remove((x, y))

            if x is None and y is None:
                listEdger.append([vertex, vertex, 'D'])
            else:
                listEdger.append([vertex, (x, y), 'D'])


    return listEdger

#10 execuções calcular desvio, media, pior e melhor
files = [
    # 'albano' #esse OK
    # 'fu', #esse OK
    # 'instance_01_2pol', #esse OK
    # 'instance_01_4pol', #esse OK
    # 'instance_01_5pol', #esse OK
    # 'instance_01_9pol', #esse OK
    # 'dighe1', #esse OK
    # 'shapes2' # esse OK
     'rco3' # esse OK
]

opcoes = {'pop': [500, 1000, 5000], 'elite': [.3, .25, .20], 'mut': [.05, .10, .15]}
op = []
for i in opcoes['pop']:
    for j in opcoes['elite']:
        for k in opcoes['mut']:
            op.append((i, j, k))
# toolbox of GA
toolbox = base.Toolbox()
# Class Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Representation Individual
creator.create("Individual", list, fitness=creator.FitnessMin)
tipo = ['packing']
if __name__ == "__main__":
    for t in tipo:
        for f in files:
            file = open(f"../../datasets/particao_arestas/ejor/{t}/{f}.txt").read(
            ).strip().split('\n')

            edges = []
            if file:
                edges = preProcess(file)
            # Generate Individual
            toolbox.register("indices", genIndividuoRK, edges)
            # initializ individual
            toolbox.register(
                "individual",
                tools.initIterate,
                creator.Individual,
                toolbox.indices
            )
            # plotar(individuo, 'iteracao')
            # exit(0)
            # Generate Population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Objective Function
            toolbox.register("evaluate", evalCut)
            # function to execute map
            toolbox.register("map", map)

            hof = None
            jaja = 0
            for k in op:
                jaja = jaja + 1
                qtd = 1
                if True:
                    file_write = None
                # with open(f"resultados/brkga/{t}/{f}_[{k[0]},{k[1]},{k[2]}].txt", mode='w+') as \
                #         file_write:
                    print("BRKGA:", file=file_write)
                    print(file=file_write)
                    for i in range(qtd):

                        print(f"Execução {jaja}:", file=file_write)
                        print(
                            f"Parametros: P={k[0]}, Pe={k[1]}, Pm={k[2]}, pe=0.7, Parada=150",
                            file=file_write
                        )
                        iteracao = None
                        with timeit(file_write=file_write):
                            iteracao = main(
                                P=k[0],
                                Pe=k[1],
                                Pm=k[2],
                                file=file_write
                            )
                        # print("Individuo:", decode(iteracao[2][0]), file=file_write)
                        print("Fitness: ", iteracao[2][0].fitness.values[0], file=file_write)
                        # print("Gens: ", iteracao[3], file=file_write)
                        # print("Inds: ", iteracao[4], file=file_write)
                        print(file=file_write)
                        dadosExel.append((iteracao[2][0].fitness.values[0], tempoExecution))
                        plotar(iteracao[2][0], f"rco310" + str(k[0]) + ' ' + str(k[1]) + ' ' + str(k[2]))
                        fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
                        fig1.set_size_inches((10, 10))
                        gens, inds = iteracao[3], iteracao[4]
                        f1_axes.set_ylabel("Valor do Melhor Individuo")
                        f1_axes.set_xlabel("Gerações")
                        f1_axes.grid(True)
                        f1_axes.set_xlim(0, gens[-1])
                        f1_axes.set_ylim(inds[-1] - 10, inds[0] + 10)
                        f1_axes.plot(gens, inds, color='blue')
                        plt.close()
            exit(0)
