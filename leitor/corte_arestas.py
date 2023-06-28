"""."""
# from random import shuffle
from pprint import pprint
# from shutil import rmtree
# from os import mkdir
# import matplotlib.pyplot as plt
print, pprint = pprint, print


class Grafo:
    """."""

    def __init__(self, v, e):
        """."""
        self._g = {}
        self.v = v
        self.e = e

    def addAresta(self, v1: (int or float), v2: (int or float), p: (int or float) = 0):
        """."""
        if not (v1 in self._g.keys()):
            self._g[v1] = {}
        self._g[v1][v2] = p

    @property
    def g(self):
        """."""
        return self._g

    def __repr__(self):
        """."""
        return str(self._g)


def dist2pt(x1, y1, x2, y2):
    """."""
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1 / 2)


def ptmed(x1, y1, x2, y2):
    """."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def corta(g: dict, pi: (int or float), mi: (int or float), mostra=True):
    """Corta as arestas do grafo.

    args:
    g -> grafo
    pi -> velocidade do corte
    mi -> velocidade do deslocamento
    """
    # somatorio = []
    arestas = [(i, j) for i in g.keys() for j in g[i]]
    arestasCortadas = []
    cortadas = []
    # cores = ['red', 'yellow']
    # shuffle(arestas)
    # rmtree('plot')
    # mkdir('plot')
    # plt.xlim(0, 40)
    # plt.ylim(0, 40)
    for i in range(len(arestas)):
        if not(arestas[i] in arestasCortadas or (arestas[i][1], arestas[i][0]) in arestasCortadas):
            # x, y = [], []
            # x1, y1 = [], []
            cortadas.append([arestas[i]])
            if i == 0:
                # somatorio.append((g[arestas[i][0]][arestas[i][1]])/pi)
                cortadas[-1].append((g[arestas[i][0]][arestas[i][1]]) / pi)
            else:
                if arestas[i - 1][1] == arestas[i][0]:
                    # somatorio.append((g[arestas[i][0]][arestas[i][1]])/pi)
                    cortadas[-1].append((g[arestas[i][0]][arestas[i][1]]) / pi)
                else:
                    # somatorio.append(
                    #     (dist2pt(*arestas[i-1][1], *arestas[i][0]))/mi +
                    #     (g[arestas[i][0]][arestas[i][1]])/pi)
                    # x1.append(arestas[i-1][1][0])
                    # y1.append(arestas[i-1][1][1])
                    # x1.append(arestas[i][0][0])
                    # y1.append(arestas[i][0][1])
                    # plt.plot(x1, y1, '-*', color=cores[1])
                    cortadas[-1].append(
                        (dist2pt(*arestas[i - 1][1], *arestas[i][0])) / mi)
                    cortadas[-1].append((g[arestas[i][0]][arestas[i][1]]) / pi)
            # x.append(arestas[i][0][0])
            # y.append(arestas[i][0][1])
            # x.append(arestas[i][1][0])
            # y.append(arestas[i][1][1])
            arestasCortadas.append(arestas[i])
            # plt.plot(x, y, '-*', color=cores[0])
            # plt.annotate(str(len(arestasCortadas)), ptmed(
            #     *arestas[i][0], *arestas[i][1]))
            # plt.savefig(f"plot/g{len(arestasCortadas)}.png")
    # print(*cortadas[0], *cortadas[1])
    out = str(len(cortadas))
    if mostra:
        pprint(len(cortadas))
    for i in cortadas:
        if mostra:
            pprint(*i[0][0], *i[0][1])
        out += f'\n{i[0][0][0]} {i[0][0][1]} {i[0][1][0]} {i[0][1][1]}'
    # pprint(out)
    # print(len(arestasCortadas))
    # print(sum(somatorio))
    return out
    # return sum(somatorio)


def cortar(ent, mostra=True):
    """."""
    # print(ent)
    g = Grafo(*ent[0])
    for i in range(1, int(g.e)):
        ent2 = [float(j) for j in ent[i]]
        g.addAresta((ent2[0], ent2[1]), (ent2[2], ent2[3]), ent2[4])
    return corta(g.g, 1, 5, mostra)


if __name__ == "__main__":
    ent = [[int(i) for i in input().split()]]
    while True:
        try:
            ent.append([float(i) for i in input().split()])
        except EOFError:
            break
    cortar(ent)
