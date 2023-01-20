import pandas as pd
import pickle as pkl
import networkx as nx
import matplotlib
from scipy import stats

mapP2UNIPROT = pkl.load(open("data/mapP2UNIPROT.pkl", 'rb'))
mapP2PrefName = pkl.load(open("data/mapP2PrefName.pkl", 'rb'))
mapGOBP2PrefName = pkl.load(open("data/mapGOBP2PrefName.pkl", 'rb'))
mapC2PrefName = pkl.load(open("data/mapC2PrefName.pkl", 'rb'))
mapM2PrefName = pkl.load(open("data/mapM2PrefName.pkl", 'rb'))
relGOBP2lP = pkl.load(open("data/relGOBP2lP.pkl", 'rb'))
relP2lGOBP = pkl.load(open("data/relP2lGOBP.pkl", 'rb'))
GraphPPi = pkl.load(open("data/GraphPPi.pkl", 'rb'))
relC2lP = pkl.load(open("data/relC2lP.pkl", 'rb'))
relP2lC = pkl.load(open("data/relP2lC.pkl", 'rb'))
relC2M = pkl.load(open("data/relC2M.pkl", 'rb'))

def getPfromlGOBP(lGOBP):
    s = set()
    for GOBP in lGOBP: 
        GOBP = int(GOBP[3:])
        if GOBP not in relGOBP2lP: continue
        s.update(relGOBP2lP[GOBP]) 
    return s
    
def dataframe_lP_Prefname(lP):
    l = []
    for P in lP: l.append((P, mapP2UNIPROT[P], mapP2PrefName[P]))
    return  pd.DataFrame(l, columns = ["Id", "UNIPROT", "Pref name"])


def drawGraph(G):
    nx.draw(G, labels={P:mapP2PrefName[P] for P in G.nodes()})

def ConnectProteingraphPPi(G):
    lP = G.nodes()
    lPMinG = set([])
    UnconnectableP=[]
    for p1 in lP:
        if len(GraphPPi[p1])==0:
            UnconnectableP.append(p1)
            lPMinG = lPMinG.union(set([p1]))
            continue
        try:
            score, path = nx.multi_source_dijkstra(GraphPPi, [p2 for p2 in lP if p2 != p1], target=p1, cutoff=None, weight='weight')
            lPMinG = lPMinG.union(set(path))
        except:
            UnconnectableP.append(p1)
            lPMinG = lPMinG.union(set([p1]))
    Gmin=GraphPPi.subgraph(lPMinG)
    # try to connect separated subgraph
    while nx.is_connected(Gmin) == False:
        lg = sorted([[len(g), g]for g in nx.connected_components(Gmin)])
        beastScore, beastPath = 100000, []
        for i in range(len(lg)):
            allotherNodes = [n for n in lPMinG if n not in lg[i][1]]
            for n in lg[i][1]:
                try:
                    score, path = nx.multi_source_dijkstra(GraphPPi, allotherNodes, target=n, cutoff=None, weight='weight')
                    if score < beastScore: beastScore, beastPath = score, path
                except:
                    pass
        for n in beastPath : lPMinG.add(n)
        Gmin=GraphPPi.subgraph(lPMinG)
        if len(beastPath)==0: break
    
    return GraphPPi.subgraph(list([p for p in lPMinG if p not in lP])+list(lP))

def dataframe_betweenness_centrality(G):
    l = [(P, mapP2UNIPROT[P], mapP2PrefName[P], s) for P, s in nx.betweenness_centrality(G).items()]
    return  pd.DataFrame(l, columns = ["Id", "UNIPROT", "Pref name", 'betweenness_centrality'])
        
def dataframe_eigenvector_centrality(G):
    l = [(P, mapP2UNIPROT[P], mapP2PrefName[P], s) for P, s in nx.eigenvector_centrality(G).items()]
    return  pd.DataFrame(l, columns = ["Id", "UNIPROT", "Pref name", 'eigenvector_centrality']) 
    
    
def enrichissementGOBPfromP(lIn1, sortCriteriaForLimit="fold", minScoreInSample=0, minsample_freq=0, minFold=2, maxpValueBinomial=0.05, type_pValue='greater'):
    lIn1 = set(lIn1)
    lBg = list(relGOBP2lP.keys())
    lIn2F = set()
    for P in lIn1: 
        if P not in relP2lGOBP: continue
        lIn2F.update(relP2lGOBP[P])

    nbBg = len(lBg)
    dIdw = {Id:1 for Id in lIn1}
    nbSample = len(lIn1)
    lEnrich = []
    for GOBP in lIn2F:
        nbTotal = 0
        for T in relGOBP2lP[GOBP]:
            if T in dIdw: nbTotal+=dIdw[T]
            else:         nbTotal+=1
        bg_freq = nbTotal/nbBg
        expected = bg_freq*nbSample
        lIn1Sample = list(relGOBP2lP[GOBP].intersection(lIn1))
        scoreInSample = sum([dIdw[P] for P in lIn1Sample])
        sample_freq = scoreInSample/nbSample
        fold = sample_freq/bg_freq
        pValueCBinomial = stats.binom_test(scoreInSample, n=nbTotal, p=expected/nbTotal, alternative=type_pValue)
        if pValueCBinomial>maxpValueBinomial or scoreInSample<minScoreInSample or sample_freq*100<minsample_freq or fold<minFold: continue
        lEnrich.append([GOBP, mapGOBP2PrefName[GOBP], round(fold, 1), pValueCBinomial,  nbTotal, scoreInSample, [mapP2PrefName[P] for P in lIn1Sample]])  

    if   sortCriteriaForLimit == 'scoreInSample':  lEnrich.sort(key=lambda x: x[6], reverse=True)
    else:                                          lEnrich.sort(key=lambda x: x[2], reverse=True)

    return pd.DataFrame(lEnrich, columns = ['Target Id', 'Target Name', 'fold', 'pValue Binomial', 'nbTotal', 'scoreInSample', 'lIn1Sample'])
    
def enrichissementCfromP(lIn1, sortCriteriaForLimit="fold", minScoreInSample=0, minsample_freq=0, minFold=2, maxpValueBinomial=0.05, type_pValue='greater'):
    lIn1 = set(lIn1)
    lBg = list(relC2lP.keys())
    lIn2F = set()
    for P in lIn1: 
        if P not in relP2lC: continue
        lIn2F.update(relP2lC[P])

    nbBg = len(lBg)
    dIdw = {Id:1 for Id in lIn1}
    nbSample = len(lIn1)
    lEnrich = []
    for C in lIn2F:
        nbTotal = 0
        for T in relC2lP[C]:
            if T in dIdw: nbTotal+=dIdw[T]
            else:         nbTotal+=1
        bg_freq = nbTotal/nbBg
        expected = bg_freq*nbSample
        lIn1Sample = list(relC2lP[C].intersection(lIn1))
        scoreInSample = sum([dIdw[P] for P in lIn1Sample])
        sample_freq = scoreInSample/nbSample
        fold = sample_freq/bg_freq
        pValueCBinomial = stats.binom_test(scoreInSample, n=nbTotal, p=expected/nbTotal, alternative=type_pValue)
        if pValueCBinomial>maxpValueBinomial or scoreInSample<minScoreInSample or sample_freq*100<minsample_freq or fold<minFold: continue
        lEnrich.append([C, mapC2PrefName[C], round(fold, 1), pValueCBinomial,  nbTotal, scoreInSample, [mapP2PrefName[P] for P in lIn1Sample]])  

    if   sortCriteriaForLimit == 'scoreInSample':  lEnrich.sort(key=lambda x: x[6], reverse=True)
    else:                                          lEnrich.sort(key=lambda x: x[2], reverse=True)

    return pd.DataFrame(lEnrich, columns = ['Target Id', 'Target Name', 'fold', 'pValue Binomial', 'nbTotal', 'scoreInSample', 'lIn1Sample'])

def getSideEventC(C):
    if C not in relC2M: 
        print('no side event repported')
        return 
    l = []
    for M in relC2M[C]: l.append((M, mapM2PrefName[M]))
    return pd.DataFrame(l, columns = ['Id', 'side event'])
