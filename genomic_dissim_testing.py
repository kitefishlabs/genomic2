'''
from genomic7 import *
from bregman.suite import *
import scipy.spatial.distance as spat

genex = GenomicExplorer2('/Users/kfl/dev/git/public_projects/genomic2', 'viola_1.wav', 'chinahit_2.wav', size=30)
genex.iterate(30)

'''
targ = genex.corpus.convert_corpus_to_tagged_array('m13', tag = -1, map_flag=True)[:,1:]
targa = genex.corpus.convert_corpus_to_tagged_array('p6', tag = -1, map_flag=True)[:,1:3]


pop = genex.corpus.convert_corpus_to_array_by_cids('m13', cids = genex.current_roster, map_flag=True)[:,1:]
popa = genex.corpus.convert_corpus_to_array_by_cids('p6', cids = genex.current_roster, map_flag=True)[:,1:3]

mean = ma.masked_where(pop==0.0,pop).mean(axis=0).data
meana = ma.masked_where(popa==0.0,popa).mean(axis=0).data

popmean = np.r_[pop, np.atleast_2d(mean)]
popmeana = np.r_[popa, np.atleast_2d(meana)]


pop = np.r_[targ, pop]
popa = np.r_[targa, popa]


pop_spatdist = spat.pdist(pop, 'seuclidean')
popa_spatdist = spat.pdist(popa, 'seuclidean')

mpop_spatdist = spat.pdist(popmean, 'seuclidean')
mpopa_spatdist = spat.pdist(popmeana, 'seuclidean')

pop_squaredist = spat.squareform(pop_spatdist)
popa_squaredist = spat.squareform(popa_spatdist)

mpop_squaredist = spat.squareform(mpop_spatdist)
mpopa_squaredist = spat.squareform(mpopa_spatdist)

m_tscores = [pop_squaredist[0,indiv] for indiv in range(1,31)]
a_tscores = [popa_squaredist[0,indiv] for indiv in range(1,31)]

m_mscores = [mpop_squaredist[30,indiv] for indiv in range(0,30)]
a_mscores = [mpopa_squaredist[30,indiv] for indiv in range(0,30)]

# total_scores = [
# 	[i, (
# 		(1.0*(m_tscores[i]/max(m_tscores))) + 
# 		(a_tscores[i]/max(a_tscores)) + 
# 		(1.0*(m_mscores[i]/max(m_mscores))) + 
# 		(a_mscores[i]/max(a_mscores))
# 	)] for i in range(30)
# ]

# imagesc(pop.T)
# imagesc(popa.T)
# imagesc(popmean.T)
# imagesc(popmeana.T)

# imagesc(pop_squaredist)
# imagesc(popa_squaredist)

# imagesc(mpop_squaredist)
# imagesc(mpopa_squaredist)




# imagesc(spat.squareform(spat.pdist(np.array([m_tscores, a_tscores, m_mscores, a_mscores]).T, 'seuclidean')))
# plt.plot(np.array([m_tscores, a_tscores, m_mscores, a_mscores]).T.sum(axis=1))


ordering = sorted([[i, m_tscores[i], a_tscores[i], m_mscores[i], a_mscores[i]] for i in range(30)], key = lambda row: (row[1]+row[2]+row[3]+row[4]), reverse=False)