import sc, random, contextlib, wave, os, math
import shlex, subprocess, signal
import nrt_osc_parser_genomic

from itertools import chain, izip
import numpy as np
import numpy.ma as ma
import scipy.signal
import matplotlib.pyplot as plt
import scipy.spatial.distance as spat

from corpusdb import *

# generator class for weighted random numbers
#
# Pass in one or the other:
# - weights: custom weights array
# - size: size of "standard" weights array that algo should make on its own
#
# call next to actually make the random selection
#
class RandomGenerator_8Bit(object):

	def __init__(self, initval=-1):
		if initval >= 0:
			self.val = initval
		else:
			self.val = random.randint(0,256)
		
	def next(self, scale=1.0):
		self.val = random.randint(0,256)

	def __call__(self): return self.next()


class GenomicExplorer2:

	def __init__(self, anchor, sourcesound, targetsound, subdir=None, out_dir='out', psize=50, report_interval=20, mut_prob=0.01, stop_slope=0.000001):
				
		self.anchor = anchor
		if subdir is not None:
			self.srcpath = os.path.join(anchor, 'snd', subdir, sourcesound)
			self.targpath = os.path.join(anchor, 'snd', subdir, targetsound)
		else:
			self.srcpath = os.path.join(anchor, 'snd', sourcesound)
			self.targpath = os.path.join(anchor, 'snd', targetsound)
		self.out_dir = out_dir
		
		self.current_generation = 1
		self.gspec = [
			ParameterGeneSequence(0, 'efx_thru_mn', ['dummy'], [[0.0,1.0]]),
			ParameterGeneSequence(1, 'efx_gain_mn', ['gain'], [[0.0,1.0]]),
			
			ParameterGeneSequence(2, 'efx_freqshift_mn', ['frequency', 'phase', 'gain'], [[0,1000],[0.2,6.2831853072],[0.0,1.0]]),
			ParameterGeneSequence(3, 'efx_pitchshift_mn', ['windowsize', 'pitchratio', 'pitchdisp', 'timedisp', 'gain'], [[0.01,0.5],[0.5,2.0],[0,1],[0,0.5],[0.2,2.0]]),
			ParameterGeneSequence(4, 'efx_monograin_mn', ['windowsize', 'grainrate', 'winrandperc'], [[0.01,0.05],[1.0,100.0],[0,1],[0.0,1.0]]),
			ParameterGeneSequence(5, 'efx_comb_mn', ['delay', 'decay', 'gain'], [[0.0001,0.05],[0.2,2.0],[0.0,1.0]]),
			
			ParameterGeneSequence(6, 'efx_clipdist_mn', ['mult', 'clip', 'gain'], [[0.0,10.0],[0.2,1.0],[0.0,1.0]]),
			ParameterGeneSequence(7, 'efx_softclipdist_mn', ['mult', 'sclip', 'gain'], [[0.0,5.0],[0.2,1.0],[0.0,1.0]]),
			ParameterGeneSequence(8, 'efx_decimatordist_mn', ['rate', 'bits', 'gain'], [[44100,441],[0,24],[0.0,1.0]]),
			ParameterGeneSequence(9, 'efx_smoothdecimatordist_mn', ['rate', 'smooth', 'gain'], [[44100,441],[0.0,1.0],[0.0,1.0]]),
			ParameterGeneSequence(10, 'efx_crossoverdist_mn', ['amp', 'smooth', 'gain'], [[0.0,10.0],[0.2,1.0],[0.0,1.0]]),
			ParameterGeneSequence(11, 'efx_sineshaperdist_mn', ['amp', 'smooth', 'gain'], [[0.0,10.0],[0.2,1.0],[0.0,1.0]]),

			ParameterGeneSequence(12, 'efx_ringz_mn', ['freq', 'decay', 'gain'], [[10000,40],[0,1],[0.2,1.0]]),
			ParameterGeneSequence(13, 'efx_resonz_mn', ['freq', 'bwr', 'gain'], [[10000,40],[0.01,2],[0.2,1.0]]),
			ParameterGeneSequence(14, 'efx_formlet_mn', ['freq', 'atk', 'decay', 'gain'], [[0,100],[0.001,0.01],[0.001,0.01],[0.2,1.0]]),
			ParameterGeneSequence(15, 'efx_bmoog_mn', ['freq', 'qval','mode', 'saturation', 'gain'], [[10000,40],[0.01,2],[0,3],[0,1],[0.2,1.0]]),
			


			ParameterGeneSequence(16, 'efx_lopass_mn', ['center', 'gain'], [[10000,40],[0.2,1.0]]),
			ParameterGeneSequence(17, 'efx_hipass_mn', ['center', 'gain'], [[40,10000],[0.2,1.0]]),
			ParameterGeneSequence(18, 'efx_blowpass_mn', ['center', 'rq', 'gain'], [[10000,40],[0.001, 1.0],[0.2,1.0]]),
			ParameterGeneSequence(19, 'efx_blowpass4_mn', ['center', 'rq', 'gain'], [[10000,40],[0.001, 1.0],[0.2,1.0]]),
			ParameterGeneSequence(20, 'efx_bhipass_mn', ['center', 'rq', 'gain'], [[40,10000],[0.001, 1.0],[0.2,1.0]]),
			ParameterGeneSequence(21, 'efx_bhipass4_mn', ['center', 'rq', 'gain'], [[40,10000],[0.001, 1.0],[0.2,1.0]]),
			ParameterGeneSequence(22, 'efx_bpeakeq_mn', ['center', 'rq', 'db', 'gain'], [[100,2000],[0.001, 1.0],[-24,0],[0.2,1.0]]),
			ParameterGeneSequence(23, 'efx_blowshelf_mn', ['center', 'rq', 'db', 'gain'], [[10000,40],[0.001, 1.0],[-24,0],[0.2,1.0]]),
			ParameterGeneSequence(24, 'efx_bhishelf_mn', ['center', 'rq', 'db', 'gain'], [[40,10000],[0.001, 1.0],[-24,0],[0.2,1.0]]),
			ParameterGeneSequence(25, 'efx_bbandstop_mn', ['center', 'rq', 'gain'], [[100,2000],[0.001, 1.0],[0.2,1.0]]),
			ParameterGeneSequence(26, 'efx_ballpass_mn', ['center', 'rq', 'gain'], [[100,2000],[0.001, 1.0],[0.2,1.0]]),

			ParameterGeneSequence(27, 'efx_pvpartialsynth_mn', ['pcut', 'gain'], [[0,1.0],[0.2,1.0]]),
			ParameterGeneSequence(28, 'efx_pvmagsmear_mn', ['bins', 'gain'], [[0,1024],[0.2,1.0]]),
			ParameterGeneSequence(29, 'efx_pvbrickwall_mn', ['wipe', 'gain'], [[-1,1],[0.2,1.0]]),
			ParameterGeneSequence(30, 'efx_pvmagabove_mn', ['thresh', 'gain'], [[0,1],[0.2,1.0]]),
			ParameterGeneSequence(31, 'efx_pvmagbelow_mn', ['thresh', 'gain'], [[0,1],[0.2,1.0]]),
			ParameterGeneSequence(32, 'efx_pvmagclip_mn', ['thresh', 'gain'], [[0,1],[0.2,1.0]]),
			ParameterGeneSequence(33, 'efx_pvphaseshift_mn', ['shift', 'gain'], [[40,10000],[0.2,1.0]]),
			ParameterGeneSequence(34, 'efx_pvphaseshifti_mn', ['shift', 'gain'], [[0,6.2831853072],[0.2,1.0]]),
			ParameterGeneSequence(35, 'efx_hipass_mn', ['center', 'gain'], [[0,6.2831853072],[0.2,1.0]])
			]

		# create the corpus and add the source and target sounds
		self.corpus = corpusdb.CorpusDB(anchor)
		
		self.source_id = self.add_and_analyze_parent(sourcesound, tag=0)  	# SOURCE = 0
		self.target_id = self.add_and_analyze_parent(targetsound, tag=-1)	# TARGET = -1

		self.current_roster = dict()

		self.mutation_prob = mut_prob
		self.depth = 10
		self.pop_size = psize
 		# 'alpha', 'c_delay', 'c_decay', 'beta', 'd_mult', 'gamma', 'ms_bins'
		# self.rawtable, self.rawmaps, 
		self.history, self.dists, self.pool_means, self.pool_stdevs = dict(), dict(), dict(), dict()

		self.reporting_interval = report_interval
		self.stopping_slope = stop_slope
		self.running_avg_mean_stdevs = dict()
		self.stopping_crit_min_gens = 5
		self.init_population()

	
	def init_population(self):
		self.population = []
		for n in range(self.pop_size):
			self.population += [Genome(self.gspec)] #random seed
			# convert genome params into child node + analyze
			#self.add_and_analyze_child(n)
			self.add_and_analyze_child_grid(n)
		#print self.population
		# self.compare_all_individuals(aflag=True)
	
	# Iterate over population and mutate with some probibility
	def mutate_pop(self):
		mutated = []
		for indiv in range(1, len(self.population)):
			#if random.random() < self.mutation_prob:
				# print "indiv: ", indiv
			self.population[ indiv ].mutate()
			mutated += [indiv] # mark an indiv. that has mutated
			self.do_update_cascade(indiv)
			mutated += [indiv]
		self.history[self.current_generation] = mutated
				
	# This is performed once per mutation
	# - only affects the individual being mutated
	def do_update_cascade(self, index, clearedits=False):
		if clearedits is True:
			self.population[ index ].edits = 0
		else:
			self.population[ index ].edits += 1
		# self.add_and_analyze_child(index, tag=self.current_generation)
		self.add_and_analyze_child_grid(index, tag=self.current_generation)
	
	# Mate with single-point crossover
	def mate(self, a, b, kill_index):
		offspring = None		
		# print "--------------mate2..."
		# print [a,b,kill_index]
		if random.random() < 0.5:
			offspring = self.population[a].values[:]
		else:
			offspring = self.population[b].values[:]
		# basic random gene(wise) selection from 2 parents
		for i in range(len(offspring)):
			if random.random() < 0.5:
				offspring[i] = self.population[a].values[i]
			else:
				offspring[i] = self.population[b].values[i]
		# replace the killed individual with our new individual
		self.population[kill_index] = Genome(self.gspec,offspring)
		self.do_update_cascade(kill_index, True)

	# Mate with 2-point (cyclic) crossover, use offspringB
	def mate2(self, a, b, kill_index):
		offspring = None
		# print "--------------mate2..."
		# print [a,b,kill_index]
		if random.random() < 0.5:
			offspringA = self.population[a].values[:]
			offspringB = self.population[b].values[:]
		else:
			offspringA = self.population[b].values[:]
			offspringB = self.population[a].values[:]
		# print (offspringA, offspringB, (len(offspringA) - 1))
		l = len(offspringA) - 1
		cuts = sorted([random.randint(0,l), random.randint(0,l)])
		# print cuts
		temp = offspringA[cuts[0]:cuts[1]]
		offspringA[cuts[0]:cuts[1]] = offspringB[cuts[0]:cuts[1]]
		offspringB[cuts[0]:cuts[1]] = temp
		# print (offspringA, offspringB)
		# replace the killed individual with our new individual
		self.population[kill_index] = Genome(self.gspec,offspringA)
		self.do_update_cascade(kill_index, True)

	# FITNESS FUNCTION:
	# 1. calculate 3 values for all members:
	#	A. using amplitude/power vals, pop. member -> target indiv.
	#	B. using MFCCs, pop. member -> target indiv.
	#	C. using MFCCs, pop. member -> pop. MEAN
	def compare_pop_to_target(self, depth, verb=False):
		
		targ = self.corpus.convert_corpus_to_tagged_array('m13', tag = -1, map_flag=True)[:,1:]
		targa = self.corpus.convert_corpus_to_tagged_array('p6', tag = -1, map_flag=True)[:,1:3]


		pop = self.corpus.convert_corpus_to_array_by_cids('m13', cids = self.current_roster, map_flag=True)[:,1:]
		popa = self.corpus.convert_corpus_to_array_by_cids('p6', cids = self.current_roster, map_flag=True)[:,1:3]

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

		m_tscores = [pop_squaredist[0,indiv] for indiv in range(1,(self.pop_size+1))]
		a_tscores = [popa_squaredist[0,indiv] for indiv in range(1,(self.pop_size+1))]

		m_mscores = [mpop_squaredist[self.pop_size,indiv] for indiv in range(self.pop_size)]
		a_mscores = [mpopa_squaredist[self.pop_size,indiv] for indiv in range(self.pop_size)]

		ordered = sorted([[i, m_tscores[i], a_tscores[i], (1.0-(m_mscores[i]/max(m_mscores)))] for i in range(self.pop_size)], key = lambda row: (row[1]+row[2]+row[3]), reverse=False)

		print "ORDERED: ", ordered

		return ordered[:depth], ordered[(-1*depth):]


	def reproduce(self, depth=25):
		
		duplicates, kills  = self.compare_pop_to_target(depth)
		# for dup in duplicates:
		# 	self.render_individual(dup[0], 'gen'+str(self.population[0].age))
		# print 'depth: ', depth
		# depth # of times: choose 2 random parents to mate and overwrite replacement in unfit individual's slot
		for n in range(depth):
			# print 'num. duplicates: ', len(duplicates)
			aidx = duplicates[ random.randint(0, depth-1) ][0]
			bidx = duplicates[ random.randint(0, depth-1) ][0]
			
			kidx = kills[ random.randint(0, depth-1) ][0]
			# print "ASDASDASD"
			
			self.mate2(aidx, bidx, kidx)
	
	def age_pop(self):
		for i in range(len(self.population)): self.population[i].age += 1
		self.current_generation += 1
	
	def iterate(self, iters=1):
		# sc.quit()
		for iter in range(iters):
			self.age_pop()
			self.mutate_pop()
# 			self.crossover()
			if (iter%self.reporting_interval)==0:
				# print self.population[10].age
				self.reproduce(self.depth)
				#self.collect_population_data()
				# res = self.check_for_stopping_conditions()
				# if (res == 1) and (self.population[0].age > self.reporting_interval):
				# 	return
	
	def print_all_individuals(self):
		print '== pop ==========================='
		for g in self.population: print g
	
	def start_sc(self):
		try:
			sc.start(verbose=1, spew=1, startscsynth=1)
		except OSError: # in case we've already started the synth
			print 'QUIT!'
			sc.quit()
		print 'sfpaths: ', self.sfpaths
		for i, sfpath in enumerate(self.sfpaths):
			bnum = sc.loadSnd(os.path.basename(sfpath), wait=False)
			print 'bnum: ', bnum
			self.sfinfos[i]['bnum'] = bnum
		return 1
	
	
	def add_and_analyze_parent(self, sourcefile, subdir=None, tag=0, verb=False):

		self.sourcesnd = self.corpus.add_sound_file(sourcefile, subdir=subdir, verb=verb)
		sourceid = self.sourcesnd.sfid

		self.corpus.analyze_sound_file(self.sourcesnd, sourceid, outwav=False, topoflag=0, verb=verb)
		sfdur = self.corpus.sftree.nodes[sourceid].duration
		sfid = self.corpus.add_sound_file_unit(sourceid, onset=0, dur=sfdur, tag=tag)

		# print "source id should == sfid: ", (sourceid == sfid)

		self.corpus.segment_units(sfid)
		
		return sfid


	def add_and_analyze_child(self, index, tag=1, verb=False):

		genome = self.population[index]
		print ""
		print genome.realvalues[:4]
		slots = [(int(val)%36) for val in genome.realvalues[:4]]
		revised = unique_listed(slots)
		# print "SLOTS: ", revised

		# print "--- ", genome.pgsequences
		# print revised
		pgseqs = [genome.pgsequences[id] for id in revised]
		# print pgseqs
		synthlist = [pgs.efxsynth for pgs in pgseqs]
		paramlist = [pgs.params for pgs in pgseqs]
		# print "---------"
		# print synthlist
		# print paramlist

		for n,slot in enumerate(revised):
			# print ''
			# print slot, " + ", n
			# print genome.realvalues
			valchunk = genome.realvalues[(4+genome.alist[slot]):(4+genome.alist[slot+1])]
			# print paramlist[n]
			# print valchunk
			paramlist[n] = ['inbus', 10, 'outbus', 11] + list(chain.from_iterable(izip(paramlist[n], valchunk)))
			#	 print "param list: ", paramlist


		child_node = self.corpus.add_sound_file(filename=None,
			sfid=None,
			srcFileID=self.source_id,
			synthdef=synthlist,
			params=paramlist,
			verb=True)
		# print child_node
		self.corpus.analyze_sound_file(self.sourcesnd, child_node.sfid, outwav=True, topoflag=0, verb=verb)
		sfdur = self.corpus.sftree.nodes[child_node.sfid].duration
		sfid = self.corpus.add_sound_file_unit(child_node.sfid, onset=0, dur=sfdur, tag=self.current_generation)
		self.corpus.segment_units(sfid)

		# print sfid
		# print self.corpus.cuids_for_sfid(sfid)
		self.current_roster[index] = self.corpus.cuids_for_sfid(sfid, map_flag=True)
		# print '=========================================================', self.current_roster[index]

		return sfid, (sfid==child_node.sfid)


	def add_and_analyze_child_grid(self, index, tag=1, verb=False):

		genome = self.population[index]
		
		switches = [(int(val)%2) for val in genome.realvalues[:16]]
		slots = [(int(val)%36) for val in genome.realvalues[16:32]]
		revised = [(switches[n] * slots[n]) for n in range(16)]
		if verb:
			print "ADD AND ANALYZE CHILD _GRID_ ======"
			print genome.realvalues[:16]
			print "SLOTS: ", revised
			print "--- ", genome.pgsequences
		# print revised
		pgseqs = [genome.pgsequences[id] for id in revised]
		# print pgseqs
		synthlist = [pgs.efxsynth for pgs in pgseqs]
		paramlist = [pgs.params for pgs in pgseqs]
		# print "---------"
		# print synthlist
		# print paramlist

		for n,slot in enumerate(revised):
			# print ''
			# print slot, " + ", n
			# print genome.realvalues
			valchunk = genome.realvalues[(32+genome.alist[slot]):(32+genome.alist[slot+1])]
			# print paramlist[n]
			# print valchunk
			paramlist[n] = ['inbus', 10, 'outbus', 11] + list(chain.from_iterable(izip(paramlist[n], valchunk)))
			#	 print "param list: ", paramlist


		child_node = self.corpus.add_sound_file(filename=None,
			sfid=None,
			srcFileID=self.source_id,
			synthdef=synthlist,
			params=paramlist,
			verb=False)
		# print child_node
		# self.corpus.analyze_sound_file(self.sourcesnd, child_node.sfid, outwav=True, topoflag=0, verb=verb)
		self.corpus.analyze_sound_file(self.sourcesnd, child_node.sfid, outwav=True, topoflag=16, verb=verb)
		sfdur = self.corpus.sftree.nodes[child_node.sfid].duration
		sfid = self.corpus.add_sound_file_unit(child_node.sfid, onset=0, dur=sfdur, tag=self.current_generation)
		self.corpus.segment_units(sfid)

		# print sfid
		# print self.corpus.cuids_for_sfid(sfid)
		self.current_roster[index] = self.corpus.cuids_for_sfid(sfid, map_flag=True)
		# print '=========================================================', self.current_roster[index]

		return sfid, (sfid==child_node.sfid)



	
	"""
	COMPARE_ALL_INDIVIDUALS:
		... to individual in slot 0!
	"""
	def compare_all_individuals(self, aflag=False):
		# print self.population
		for i in range(1, len(self.population)):
 			if aflag:
 				self.analyze_individual(i)
				self.activate_raw_data(i)
#  			self.compare_individual_chi_squared(i)
			self.compare_individual(i)
		# print self.dists
		return self.dists
	
	def collect_population_data(self):
		diffs = [self.dists[k] for k in self.dists.keys()]
		print 'diffs: ', diffs
		age0 = self.population[0].age
		age1 = self.population[1].age
		print 'ages: ', age0, '|', age1
		self.pool_means[age1] = np.mean(diffs)
		self.pool_stdevs[age1] = np.std(diffs)

	def collect_population_data_resample(self):
		zero_data = np.array(self.rawmaps[0][:,1:14])
		zr0_length = zero_data.shape[0]
		diffs = []
		for indiv in range(1, len(self.population)):
			data = np.array(self.rawmaps[indiv][:,1:14])
			i_length = data.shape[0]
			if (i_length > zr0_length):
				diffs += [np.sum(np.abs(scipy.signal.signaltools.resample(data, zr0_length, window='hanning') - zero_data))]
			elif (i_length < zr0_length):
				diffs += [np.sum(np.abs(data - scipy.signal.signaltools.resample(zero_data, i_length, window='hanning')))]
			else:
				diffs += [np.sum(np.abs(data - zero_data))]
			
		diffs = np.array(diffs)
		age0 = self.population[0].age
		age1 = self.population[1].age
		print 'ages: ', age0, '|', age1
		self.pool_means[age1] = np.mean(diffs)
		self.pool_stdevs[age1] = np.std(diffs)

	def check_for_stopping_conditions(self):
		"""
		0 = continue
		1 = stop
		"""
		age0 = self.population[0].age
		stdevs_skeys = sorted(self.pool_stdevs.keys())
		self.stdevs_ordered = [self.pool_stdevs[key] for key in stdevs_skeys]
		lastNstdevs = self.stdevs_ordered[(-1*self.stopping_crit_min_gens):]
		self.running_avg_mean_stdevs[age0] = mean(lastNstdevs)
		print ">>>>>>>>>>>>>>>>> STOP??? ::: ", (abs(max(lastNstdevs) - min(lastNstdevs)) /  self.stopping_crit_min_gens)
		print ">>>>>>>>>>>>>>>>> MEAN    ::: ", mean(lastNstdevs)
		if (len(lastNstdevs) < self.stopping_crit_min_gens) or ((abs(max(lastNstdevs) - min(lastNstdevs)) /  self.stopping_crit_min_gens) > self.stopping_slope):
			print " continue ..."
			return 0
		else:
			print "**STOP**"
			return 1 # signal stop
	
	def population_realvalues_as_array(self):
		realvals = []
		for indiv in self.population:
			realvals += indiv.realvalues
		return np.array(realvals).reshape((-1,7))
	
	def population_8bitvalues_as_array(self):
		vals = []
		for indiv in self.population:
			vals += indiv.values
		return np.array(vals).reshape((-1,7))
		
	


# class ActivatorGeneSequence:

# 	def __init__(self, size=0, mappings=[]):
# 		self.size = len(mappings)
# 		self.mappings = mappings
# 		self.boundaries = [0,1]

# 		self.values = [0 for n in range(8)]
# 		self.binarystring = "00000000"
# 		self.bitlength = 8


class ParameterGeneSequence:

	def __init__(self, id, efxsynth='efx_thru_mn', params=[], ranges=[[0.0, 1.0]]):
		self.id = id
		self.size = len(params)
		self.efxsynth = efxsynth
		self.params = params
		self.boundaries = ranges

	def __repr__(self):
		return ''.join(map(str, list((self.id, self.size, self.efxsynth) + tuple(self.params) + tuple(self.boundaries))))




class Genome:

	def __init__(self, pgseq=[], values=None, verb=False):
		
		"""
		4 activator slots
		"""

		# self.numgenes = 4
		self.numgenes = 16
		
		# self.boundaries = [[0,float(len(pgseq))] for n in range(self.numgenes)]
		self.boundaries = [[0.0,2.0] for n in range(self.numgenes)] + [[0,float(len(pgseq)-1)] for n in range(self.numgenes)]

		self.pgsequences = pgseq
		self.alist = [0]
		for pgs in self.pgsequences:
			self.alist += [(pgs.size+self.alist[-1])]
			self.boundaries += [bnds for bnds in pgs.boundaries]
			self.numgenes += pgs.size
		
		if verb:
			print "+++++++++++++++++++++++++"
			print self.numgenes, " | ", self.alist, " | "
			print self.pgsequences
			print self.boundaries

		self.tratio	= 1.0		# CHECK THIS... WHY IS IT HERE/in Hertz!!! ???

		self.generators = [RandomGenerator_8Bit(-1) for n in range(self.numgenes)] # + 1 activator block
		
		if values is None:
			self.values = [gen.val for gen in self.generators]
		else:
			self.values = values
		self.bitlength = len(self.values) * 8
		self.binarystring = vals_to_binarystring(self.values)
		# print "v+b"
		# print self.values
		# print self.boundaries
		self.realvalues = [lininterp(val,self.boundaries[i]) for i,val in enumerate(self.values)]
		
		self.age = 0
		self.edits = 0
	
	def __repr__(self):
		#print '1. ', tuple(self.values)
		#print '2. ', ((self.age, self.edits) + tuple(self.values) + tuple(self.binarystring))
		return "%9i/%9i || %.6f|%.6f|%.6f|%.6f || %.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f" % ((self.age, self.edits) + tuple(self.realvalues)) # + tuple(self.binarystring)
	
	def mutate(self):
		pos = random.randint(0,(self.bitlength-1))
		# flip bit
		print 'bit flipped to: ', abs(1 - int(self.binarystring[pos],2))
		self.binarystring = substitute_char_in_string(self.binarystring, pos, abs(1 - int(self.binarystring[pos],2)))
		# recalc binary string
		self.values = binarystring_to_vals(self.binarystring)
		# print "values: ", self.values
		self.realvalues = [lininterp(val,self.boundaries[i]) for i,val in enumerate(self.values)]



def mean(arr):
	return sum([(float(val)/len(arr)) for val in arr])

def lininterp(val,bounds=[0.,1.]):
	return (((val/254.0)*(bounds[1]-bounds[0]))+bounds[0])

def substitute_char_in_string(s, p, c):
	l = list(s)
	l[p] = str(c)
	return "".join(l)

# conversion function
def midi2hz(m): return pow(2.0, (m/12.0))
	
def vals_to_binarystring(vals = [0, 0, 0, 0, 0]):
	return ''.join((("{0:08b}".format(val)) for val in vals))

# never a '0bXXX' string!
def binarystring_to_vals(binstring):
	mystring = binstring[:]
	length = len(mystring) / 8 # ignore the last digits if it doesn't chunk into 8-item substrings
	res = []
	# 	print mystring[(n*8):((n+1)*8)]
	return [int(mystring[(n*8):((n+1)*8)], 2) for n in range(length)]

def plot_one_generational_means_stdevs(pool_means1, pool_stdevs1, poolsize1):
	
	fig, ax1 = plt.subplots(nrows=1, ncols=1)
	
	timepoints1 = sorted(pool_means1.keys())
	means1 = np.array([pool_means1[tp] for tp in timepoints1])
	stdevs1 = np.array([pool_stdevs1[tp] for tp in timepoints1])
	lower_stdevs1 = np.where(np.subtract(means1, (stdevs1/2))>0, stdevs1/2, means1) # 0
	ax1.errorbar(timepoints1, means1, yerr=[lower_stdevs1, stdevs1/2], fmt='o')
		
	ax1.set_xlabel('Number of generations')
	ax1.set_ylabel('Fitness score/dissimilarity')
	plt.show()


def plot_generational_means_stdevs(pool_means1, pool_stdevs1, poolsize1, pool_means2, pool_stdevs2, poolsize2):
	
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
	
	timepoints1 = sorted(pool_means1.keys())
	means1 = np.array([pool_means1[tp] for tp in timepoints1])
	stdevs1 = np.array([pool_stdevs1[tp] for tp in timepoints1])
	lower_stdevs1 = np.where(np.subtract(means1, stdevs1)>0, stdevs1, means1) # 0
	ax1.errorbar(timepoints1, means1, yerr=[lower_stdevs1/2, stdevs1/2], fmt='o')
	#
	timepoints2 = sorted(pool_means2.keys())
	means2 = np.array([pool_means2[tp] for tp in timepoints2])
	stdevs2 = np.array([pool_stdevs2[tp] for tp in timepoints2])
	lower_stdevs2 = np.where(np.subtract(means2, stdevs2)>0, stdevs2, means2) # 0
	ax2.errorbar(timepoints2, means2, yerr=[lower_stdevs2/2, stdevs2/2], fmt='o')
	
	ax1.set_xlabel('Number of generations')
	ax2.set_xlabel('Number of generations')
	ax1.set_ylabel('Fitness score/dissimilarity')
	plt.show()
	

def unique_listed(items):
    found = set([])
    keep = []
    for item in items:
        if item not in found:
            found.add(item)
            keep.append(item)
    return keep

# if __name__=='__main__':
# 	genex = GenomicExplorer('/Users/kfl/dev/python/sc-0.3.1/genomic', 'test.wav')
# 	genex.analyze_genome(1)