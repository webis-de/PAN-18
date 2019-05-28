import os
import sys
warnMissingValues=True
warnUnknownVerifiers=True
showMoreStatistics=False
class evaluationSetup:
	def __init__(self,identifier=""):
		self.problemIdentifiers=[]
		self.verifierIdentifiers=[]
		self.numProblems=0
		self.numVerifiers=0
		self.outcomes=None
		self.truth=None
		self.trueProblemIndices=[]
		self.tresholds=[]
		self.correlations=None
		self.redundancyFactors=None
		self.identifier=identifier
		self.missing_values=[]
	def addProblem(self,identifier):
		self.problemIdentifiers.append(identifier)
		self.numProblems+=1
	def addVerifier(self,identifier):
		self.verifierIdentifiers.append(identifier)
		self.numVerifiers+=1
	def getProblemIndex(self,identifier):
		for index,ident in enumerate(self.problemIdentifiers):
			if ident[2:] == identifier[2:]:
				return index
		return None
	def getVerifierIndex(self,identifier):
		for index,ident in enumerate(self.verifierIdentifiers):
			if ident == identifier:
				return index
		return None
	def setOutcome(self,verifierIndex,problemIndex,value):
		if self.outcomes is None:
			self.outcomes=[[None for _ in self.problemIdentifiers] for _ in self.verifierIdentifiers]
		self.outcomes[verifierIndex][problemIndex] = value
	def setTrueValue(self,problemIndex,value):
		if self.truth is None:
			self.truth=[None]*self.numProblems
		self.truth[index]=value
		if value:
			self.trueProblemIndices.append(index)
	def tresholdOutcomes(self):
		vi=0
		for outcomes in self.outcomes:
			treshold=computeTreshold(outcomes, self.truth)
			self.tresholds.append(treshold)
			missing_values=[]
			self.missing_values.append(missing_values)
			for index,value in enumerate(outcomes):
				missing_values.append(value is None)
				if warnMissingValues and (value is None):
					print("No value available for setup '%s': %s/%s" % (self.identifier,self.verifierIdentifiers[vi],\
									self.problemIdentifiers[index]),file=sys.stderr)
				outcomes[index] = 0 if value is None or value <= treshold else 1
			vi += 1
def computeTreshold_fair(outcomes,truth):
	return 0.5
def computeTreshold_median(outcomes, truth):
	outcomes=sorted(outcomes)
	if len(outcomes) % 2 == 1:
		return outcomes[len(outcomes)//2]
	else:
		index=len(outcomes)//2
		return 0.5*(outcomes[index] + outcomes[index+1])
def computeTreshold_bestAccuracy(outcomes, truth):
	combined=zip((o or 0 for o in outcomes), truth)
	combined=sorted(combined, key=lambda t: t[0])
	outcomes=[t[0] for t in combined]
	truth=[t[1] for t in combined]
	correct=sum(truth)
	bestIndex=-1
	bestResult=correct
	for index,value in enumerate(truth):
		if value:
			correct -= 1
		else:
			correct += 1
		if (index == len(truth)-1 or outcomes[index] < outcomes[index+1]) and correct > bestResult:
			bestIndex=index
			bestResult=correct
	if bestIndex == -1:
		return -float('inf')
	elif bestIndex == len(truth)-1:
		return float('inf')
	else:
		return 0.5*(outcomes[bestIndex]+outcomes[bestIndex+1])
computeTreshold=computeTreshold_bestAccuracy
class obfuscation:
	def __init__(self, setup,identifier=""):
		self.setup=setup
		self.outcomes=[[None for _ in setup.problemIdentifiers] for _ in setup.verifierIdentifiers]
		self.missing_values=[[False for _ in setup.problemIdentifiers] for _ in setup.verifierIdentifiers]
		self.identifier=identifier
	def setOutcome(self,verifierIndex,problemIndex,value):
		self.outcomes[verifierIndex][problemIndex] = value
	def tresholdOutcomes(self):
		vi=0
		for outcomes,treshold in zip(self.outcomes,self.setup.tresholds):
			for index,value in enumerate(outcomes):
				missing=(value is None) or self.setup.missing_values[vi][index]
				if missing:
					if value is None and warnMissingValues:
						#print(self.setup.verifierIdentifiers[vi],self.setup.problemIdentifiers[index])
						print("No value available for setup '%s' obfuscation '%s': %s/%s" % \
						(self.setup.identifier, self.identifier, self.setup.verifierIdentifiers[vi],self.setup.problemIdentifiers[index]), \
							file=sys.stderr)
					self.missing_values[vi][index]=True
				outcomes[index] = 0 if missing or value <=treshold else 1
			vi += 1
class evaluator:
	def __init__(self,setup):
		self.setup=setup
		self.rewards=[[None for _ in setup.problemIdentifiers] for _ in setup.verifierIdentifiers]
		self.penalties=[[None for _ in setup.problemIdentifiers] for _ in setup.verifierIdentifiers]
	def setReward(self,verifierIndex,problemIndex,weight):
		self.rewards[verifierIndex][problemIndex] = weight
	def setPenalty(self,verifierIndex,problemIndex,weight):
		self.penalties[verifierIndex][problemIndex] = weight
	def calculateScore(self,obfuscation,return_max_score=False, analysis=None):
		if obfuscation.setup is not self.setup:
			raise Exception("Setups differ")
		score=0.0
		max_score=0.0
		num_missing=0
		numGood=0
		maxGood=0
		numBad=0
		maxBad=0
		possible_values=0
		avgImpact=0.0
		relevantImpacts=0.0
		for verifierIndex,originalOutcomes in enumerate(self.setup.outcomes):
			vgood=0
			vbad=0
			vscore=0.0
			obfuscatedOutcomes=obfuscation.outcomes[verifierIndex]
			for problemIndex,originalOutcome in enumerate(originalOutcomes):
				possible_values += 1
				if obfuscation.missing_values[verifierIndex][problemIndex]:
					num_missing += 1
					continue
				obfuscatedOutcome=obfuscatedOutcomes[problemIndex]
				if originalOutcome is None or obfuscatedOutcome is None:
					raise Exception("Did not expect None in the outcomes")
				#print(originalOutcome, obfuscatedOutcome)
				if originalOutcome and not obfuscatedOutcome:
					score += self.rewards[verifierIndex][problemIndex]
					vscore += self.rewards[verifierIndex][problemIndex]
					numGood += 1 if self.setup.truth[problemIndex] else 0
					vgood += 1 if self.setup.truth[problemIndex] else 0
				elif obfuscatedOutcome and not originalOutcome:
					score -= self.penalties[verifierIndex][problemIndex]
					vscore -= self.penalties[verifierIndex][problemIndex]
					numBad += 1 if self.setup.truth[problemIndex] else 0
					vbad += 1 if self.setup.truth[problemIndex] else 0
				if originalOutcome:
					max_score += self.rewards[verifierIndex][problemIndex]
					if self.setup.truth[problemIndex]:
						maxGood += 1
				else:
					if self.setup.truth[problemIndex]:
						maxBad += 1
			'''
			if analysis:
				impact=vgood-vbad
				tp=analysis.numTruePositives[verifierIndex]
				if tp == 0 or tp == analysis.numPositiveProblems:
					continue
				if impact >= 0:
					impactWeight = 1.0/tp
				else:
					impactWeight = 1.0/(analysis.numPositiveProblems - tp)
				relevantImpacts += 1
				impact *= impactWeight
				print("%s: %f ((%d-%d)*%f=%f)" % \
					(setup.verifierIdentifiers[verifierIndex], vscore, vgood, vbad, impactWeight,impact))
				avgImpact += impact
			'''
		if showMoreStatistics:
			print("%d/%d missing; got %f of %f (%d/%d good, %d/%d bad)" % (num_missing,possible_values,score,max_score,numGood,maxGood,numBad,maxBad))
		'''
		if analysis:
			print("total: %f (%f)" % (score, avgImpact/relevantImpacts))
		'''
		return max_score if return_max_score else score
class setupAnalysis1:
	def __init__(self,setup):
		self.setup=setup
		self.verifierAccuracy=[sum(a==b for (a,b) in zip(outcomes,setup.truth))/len(outcomes) for outcomes in setup.outcomes]
		self.verifierReliability=[max(0,2*score-1) for score in self.verifierAccuracy]
		self.sumReliability=sum(self.verifierReliability)
		'''
		self.problemClearness=[sum(out[i]*real for (out,real) in zip(setup.outcomes,self.verifierReliability))/self.sumReliability \
				for i in setup.trueProblemIndices]
		'''
		self.errorVectors = [[abs(out-truth) for (out,truth) in zip(outcomes, setup.truth)] for outcomes in setup.outcomes]
		#print(list(zip(setup.outcomes[1],setup.truth)))
		#print(self.errorVectors[1])
		self.means=[sum(err)/float(setup.numProblems) for err in self.errorVectors]
		self.variances=[sum( (s-mean)**2 for s in err) for (err,mean) in zip(self.errorVectors,self.means)]
		#print(self.means, self.variances)
		self.nonConstantVerifiers=[i for (i,v) in enumerate(self.variances) if v > 0]
		for i,v in enumerate(self.variances):
			if not v:
				raise Exception("Verifier %s has no variance!" % setup.verifierIdentifiers[i])
		self.correlations=[[ sum( (s1-self.means[i])*(s2-self.means[j]) for (s1,s2) in zip(self.errorVectors[i],self.errorVectors[j]))/
				(self.variances[i]*self.variances[j])**0.5 for j in self.nonConstantVerifiers] for i in self.nonConstantVerifiers]
		redundancyFactors=[]
		nextIndex=0
		for index,corr in zip(self.nonConstantVerifiers,self.correlations):
			redundancyFactors += [1.] * (index-nextIndex)
			nextIndex = index+1
			redundancyFactors.append(1.0/sum(c for c in corr if c >= 0.5))
		#redundancyFactors += [1.] * (setup.numVerifiers-nextIndex-1)
		self.redundancyFactors=redundancyFactors
		self.verifierImportance=[reliability*redundancy for (reliability,redundancy) in zip(self.verifierReliability, self.redundancyFactors)]
		self.sumImportance=sum(self.verifierImportance)
		self.problemClearness=[sum(out[i]*imp for (out,imp) in zip(setup.outcomes,self.verifierImportance))/self.sumImportance \
				for i in setup.trueProblemIndices]
		self.numPositiveProblems=float(sum(setup.truth))
		self.numTruePositives=[sum(1-abs(e) for e,t in zip(errorVector,setup.truth) if t) for errorVector in self.errorVectors]
def createSimpleEvaluator(analysis):
	setup=analysis.setup
	result=evaluator(setup)
	for verifierIndex in range(setup.numVerifiers):
		clearnessIndex=0
		for problemIndex,yes in enumerate(setup.truth):
			if yes:
				weight=analysis.problemClearness[clearnessIndex] * analysis.verifierImportance[verifierIndex]
				result.setReward(verifierIndex,problemIndex,weight)
				result.setPenalty(verifierIndex,problemIndex,weight)
				clearnessIndex += 1
			else:
				result.setReward(verifierIndex,problemIndex,0)
				result.setPenalty(verifierIndex,problemIndex,0)
	return result
def htmlOverview(setup,analysis=None,obfus=None,title=""):
	result='''<!doctype html><html><head><title>Obfuscation Safety Analysis%s</title><style type="text/css">
table {
	background-color: #EFEFEF;
	border-collapse: collapse;
}
td, th {
	border: 1px solid black;
}
.good {
	background-color: #AAFFAA;
}
.bad {
	background-color: #FFAAAA;
}
.correct {
	font-weight: bold;
}
</style></head><body>''' % (' '+title if title else '')
	if title:
		result += '<h1>'+title+'</h1>'
	result += '<p>For each problem, we note its true answer (Y/N)'
	if analysis is not None:
		result += ' together with its clearness index. For each verifier, we note its accuracy score'
	result += '.</p>'
	result += '<table><tr><th></th>'
	for verifierIndex,verifierIdentifier in enumerate(setup.verifierIdentifiers):
		#result += '<th>'+(verifierIdentifier if len(verifierIdentifier) < 5 else verifierIdentifier[:3] + verifierIdentifier[-2:])
		result += '<th'
		if len(verifierIdentifier) <= 5:
			result += '>'+verifierIdentifier
		else:
			result += ' title="%s">%s' % (verifierIdentifier,verifierIdentifier[:3] + verifierIdentifier[-2:])
		if analysis is not None:
			result += '<br>%.2f'%analysis.verifierAccuracy[verifierIndex]
		result += '</th>'
	result += '</tr>'
	def showBoolean(value,correct=None,missing=False):
		if missing:
			return '?'
		result='T' if value else 'F'
		return ('<span class="correct">%s</span>' % result) if (value == correct) else result
	clearnessIndex=0
	for problemIndex,problemIdentifier in enumerate(setup.problemIdentifiers):
		yes=setup.truth[problemIndex]
		result += '<tr><td class="%s">' % ('good' if yes else 'bad')
		result += problemIdentifier
		result += '&nbsp;(%s)' % ('Y' if yes else 'N')
		if analysis is not None and yes:
			assert(problemIndex in setup.trueProblemIndices)
			result += '&nbsp;%.2f' % analysis.problemClearness[clearnessIndex]
			clearnessIndex += 1
		result += '</td>'
		for verifierIndex in range(setup.numVerifiers):
			if setup.missing_values[verifierIndex][problemIndex]:
				result += '<td class="missing">?</td>'
				continue
			originalOutcome=setup.outcomes[verifierIndex][problemIndex]
			result += '<td'
			obfusMissing=obfus is not None and obfus.missing_values[verifierIndex][problemIndex]
			if obfus is not None:
				obfuscatedOutcome=obfus.outcomes[verifierIndex][problemIndex]
				if yes and not obfusMissing:
					if originalOutcome and not obfuscatedOutcome:
						result += ' class="good"'
					elif not originalOutcome and obfuscatedOutcome:
						result += ' class="bad"'
			result += '>'+showBoolean(originalOutcome,yes)
			if obfus is not None:
				result += '&nbsp;&rarr;&nbsp;'+showBoolean(obfuscatedOutcome,yes,obfusMissing)
			result += '</td>'
		result += '</tr>'
	result += '</table>'
	if analysis is not None:
		result += '<h2>Correlations, reliability and redundancy factors</h2>'
		result += '<p>Each cell contains the pearson correlation coefficient between the two verifiers. '
		result += 'In the top row, we note the reliability scores. In the first column, we note the corresponding redundancy factor</p>'
		result += '<table><tr><td></td>'
		for index in analysis.nonConstantVerifiers:
			identifier=setup.verifierIdentifiers[index]
			reliability=analysis.verifierReliability[index]
			result += '<th'
			if len(identifier) <= 5:
				result += '>'+identifier
			else:
				result += ' title="%s">%s' % (identifier,identifier[:3] + identifier[-2:])
			result += '<br>%.2f</th>'%reliability
		result += '</tr>'
		for index,entries in zip(analysis.nonConstantVerifiers,analysis.correlations):
			result += '<tr><th style="text-align: left;">%s&nbsp;%.2f</th>' % \
					(setup.verifierIdentifiers[index], analysis.redundancyFactors[index])
			result += ''.join('<td>%.2f</td>'%f for f in entries)
			result += '</tr>'
		result += '</table>'
	result += '</body></html>'
	return result
def readSetup(directory, truthfile,identifier=None):
	setup=evaluationSetup(identifier=(identifier or directory))
	setup.truth=[]
	with open(truthfile,'rt') as f:
		lines=(line.strip() for line in f)
		tuples=(line.split(' ') for line in lines if line)
		for tup in tuples:
			p,t = tup
			setup.addProblem(p)
			assert(t in 'YN')
			if t=='Y':
				setup.trueProblemIndices.append(len(setup.truth))
				setup.truth.append(True)
			else:
				setup.truth.append(False)
	true_ratio=float(sum(setup.truth))/setup.numProblems
	setup.outcomes=[]
	for filename in sorted(os.listdir(directory+'/'),key=lambda s: (any(c.isupper() for c in s),s)):
		#print(any(c.isupper() for c in filename),filename)
		strBefore="authorship-verification-"
		strAfter="-run-"
		if filename[:len(strBefore)] != strBefore:
			print("skip "+filename, file=sys.stderr)
			continue
		verifier=filename[len(strBefore):]
		pos=verifier.find(strAfter)
		if pos == -1:
			print("skip "+filename, file=sys.stderr)
			continue
		verifier=verifier[:pos]
		verifierIndex=setup.numVerifiers
		setup.verifierIdentifiers.append(verifier)
		setup.numVerifiers += 1
		setup.outcomes.append([None] * setup.numProblems)
		with open(directory+'/'+filename,'rt') as f:
			lines=[line.strip() for line in f]
			tuples=(line.split(' ') for line in lines if line)
			for tup in tuples:
				p,t = tup
				problemIndex=setup.getProblemIndex(p)
				if problemIndex is None:
					print(filename)
					raise Exception("Unknown problem: "+p)
				setup.setOutcome(verifierIndex,problemIndex,float(t))
	return setup
def showBoxplots(setup):
	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('TkAgg')
	plt.boxplot(setup.outcomes)
	plt.show()
def readObfuscation(setup, directory,identifier=None):
	obf=obfuscation(setup, identifier=(identifier or directory))
	for filename in os.listdir(directory):
		strBefore="authorship-verification-"
		strAfter="-run-"
		if filename[:len(strBefore)] != strBefore:
			print("skip "+filename, file=sys.stderr)
			continue
		verifier=filename[len(strBefore):]
		pos=verifier.find(strAfter)
		if pos == -1:
			print("skip "+filename,file=sys.stderr)
			continue
		verifier=verifier[:pos]
		verifierIndex=setup.getVerifierIndex(verifier)
		if verifierIndex is None:
			if warnUnknownVerifiers:
				print("Unknown verifier for obfuscation '%s': %s"%(obf.identifier,verifier),file=sys.stderr)
			continue
		with open(directory+filename,'rt') as f:
			lines=[line.strip() for line in f]
			tuples=(line.split(' ') for line in lines if line)
			for tup in tuples:
				p,t = tup
				problemIndex=setup.getProblemIndex(p)
				if problemIndex is None:
					print(filename)
					raise Exception("Unknown problem: "+p)
				obf.setOutcome(verifierIndex,problemIndex,float(t))
	obf.tresholdOutcomes()
	return obf
datasets=['pan13-test-dataset', 'pan14-test-dataset-essays', 'pan14-test-dataset-novels', 'pan15-test-dataset']
obfuscations=['obfuscation'+x for x in 'ABCDE']
originals={
"ADC64D": "bartoli15",
"41905F": "castillojuarez14",
"0E1E8B": "castro15",
"EC16E0": "castro15",
"986BCA": "frery14",
"2237B2": "ghaeini13",
"A0F0D9": "ghosh15",
"1BA270": "gillam14",
"2B4DA1": "gillam14",
"E4636D": "gillam14",
"D0E755": "grozea13",
"F4AFB2": "grozea13",
"537B85": "harvey14",
"7ED9AA": "harvey14",
"6B5892": "jankowska14",
"942692": "jankowska14",
"1568DC": "jayapal13",
"A480D5": "jayapal13",
"8BDF4A": "khonji14",
"21C631": "kocher15",
"600C8E": "kocher15",
"73EED2": "kocher15",
"4B79E0": "layton14",
"F31394": "layton14",
"A2703E": "markov15",
"BB5AFD": "markov15",
"C5A53A": "markov15",
"41F0B1": "mechti15",
"C8642F": "mezaruiz15",
"28E87B": "moreau13",
"A0FA1D": "moreau13",
"E5A674": "moreau13",
"0264BC": "moreau15",
"5E8631": "nissim15",
"6AA458": "petmanson13",
"A0EBCA": "petmanson13",
"BA1D40": "seidman13",
"A058ED": "singh14",
"FDF4A6": "solorzanosoto15",
"869F2A": "vladimir13"}
#print("%d originals" % len(originals))
def examineDataset(dataset, obfuscations):
	print(dataset+':')
	prefix='./obfuscation-evaluation/'+dataset
	setup=readSetup(prefix, prefix+'-truth.txt',identifier=dataset)
	setup.tresholdOutcomes()
	analysis=setupAnalysis1(setup)
	name=dataset
	with open('./obfuscation-evaluation/%s.html' % name, 'wt') as f:
		f.write(htmlOverview(setup, analysis, title=name))
	evaluator=createSimpleEvaluator(analysis)
	for midfix in obfuscations:
		print(dataset+'-'+midfix)
		#obf=obfuscation(setup)
		directory=prefix+'-'+midfix+'/'
		obf=readObfuscation(setup, directory,identifier=midfix)
		name=dataset+'-'+midfix
		with open('./obfuscation-evaluation/%s.html' % name, 'wt') as f:
			f.write(htmlOverview(setup, analysis, obf, title=name))
#for d in datasets:
#	examineDataset(d,obfuscations)
if __name__ == '__main__':
	import argparse
	parser=argparse.ArgumentParser(epilog='the order of the obfuscation-html-overview files should match the order of the given obfuscators')
	parser.add_argument('setupdirectory', help='the directory where the verifier\'s outputs on non-obfuscated texts can be found')
	parser.add_argument('setuptruthfile', help='the file where the true answers to the problems can be found')
	parser.add_argument('--silent-on-missing-values', help='do *not* print error messages on missing values', default=False, action='store_true')
	parser.add_argument('--silent-on-unknown-verifiers', help='do *not* print error messages on verifiers for which only '+\
		'obfuscated answers are known', default=False, action='store_true')
	parser.add_argument('--show-verifier-distributions', help='shows boxplots of the distribution of each verifier\'s confidence scores', \
		default=False, action='store_true')
	parser.add_argument('--treshold-function', help='the function to treshold the verifier\'s confidence scores', default='optimal-accuracy', \
		choices=['optimal-accuracy', 'median', 'fair'])
	parser.add_argument('--write-setup-html-overview', help='a file path to write an overview over the setup to', type=argparse.FileType('wt'))
	parser.add_argument('--obfuscation', help='a directory containing the verifier\'s outputs on obfuscated texts',action='append')
	parser.add_argument('--write-obfuscation-html-overview', help='a file path to write an overview over the obfuscator to',action='append')
	parser.add_argument('--show-latex-summary', help='outputs LaTeX snippets to summarize the overall scores', \
		default=False, action='store_true')
	parser.add_argument('--suppress_final_score', help='does NOT print the final scores', \
		default=False, action='store_true')
	'''
	parser.add_argument('--show-verifier-statistics', help='prints a table of verifier redundancy, reliability and importance scores', \
		default=False, action='store_true')
	'''
	parser.add_argument('--show-problem-clearness', help='prints a list of problem clearness scores', \
		default=False, action='store_true')
	parser.add_argument('--max-score', help='prints the maximal possible score instead of the actual score', \
		default=False, action='store_true')
	parser.add_argument('--show-more-statistics', help='shows more information to justify the obfuscator scores', \
		default=False, action='store_true')
	args=parser.parse_args()
	if args.silent_on_missing_values:
		warnMissingValues=False
	if args.silent_on_unknown_verifiers:
		warnUnknownVerifiers=False
	if args.show_more_statistics:
		showMoreStatistics=True
	setup=readSetup(args.setupdirectory, args.setuptruthfile)
	if args.show_verifier_distributions:
		showBoxplots(setup)
	tresholdDir={'optimal-accuracy': computeTreshold_bestAccuracy, 'median': computeTreshold_median, 'fair': computeTreshold_fair}
	computeTreshold=tresholdDir[args.treshold_function]
	setup.tresholdOutcomes()
	analysis=setupAnalysis1(setup)
	if args.show_latex_summary:
		print("(setupdirectory, num. problems, num. positive problems, num. negative problems, number of verifiers)")
		print("%s & %d & %d & %d & %d"%(args.setupdirectory, setup.numProblems, len([t for t in setup.truth if t]), \
			len([t for t in setup.truth if not t]), setup.numVerifiers))
		print("Verifiers: "," ".join(setup.verifierIdentifiers))
		print("Problems: "," ".join(setup.problemIdentifiers))
	if args.show_problem_clearness:
		print('internal_id\tidentifier\tclearness')
		for i,clearness in zip(setup.trueProblemIndices,analysis.problemClearness):
			print('%d\t%s\t%f' % (i,setup.problemIdentifiers[i],clearness))
	if "pan15" in args.setupdirectory and False:
		numMockVerifiers=setup.numVerifiers-40
		import matplotlib.pyplot as plt
		import matplotlib
		matplotlib.use('TkAgg')
		#plt.scatter(analysis.verifierAccuracy, analysis.redundancyFactors)
		#for (acc,red) in zip(analysis.verifierAccuracy,analysis.redundancyFactors):
		#	print("%f\t%f" % (acc,red))
		#sys.exit(0)
		#print("\n".join(str(i)+"\t"+str(c) for (i,c) in enumerate(analysis.problemClearness)))
		sys.exit(0)
		#print(analysis.verifierAccuracy)
		#print(analysis.redundancyFactors)
		print(sorted((t for t in zip(analysis.verifierAccuracy, analysis.redundancyFactors)), key=lambda t: t[1]))
		#print("%d times 1.0, %d times < 0.7" % (len([r for r in analysis.redundancyFactors if r==1.0]), \
		#		len([r for r in analysis.redundancyFactors if r<0.7])))
		#plt.scatter(analysis.problemClearness, [0.]*len(setup.trueProblemIndices), marker="|")
		plt.show()
		print("\n".join("%d %f" % tup for tup in enumerate(analysis.problemClearness)))
		'''
		print("reliability, original:")
		print("\n".join("%d %f" % (i,reliability) for (i,reliability) in enumerate(analysis.redundancyFactors[numMockVerifiers:])))
		print("reliability, mocked:")
		for i in range(40):
			ident=setup.verifierIdentifiers[i+numMockVerifiers]
			mocks=[setup.getVerifierIndex(identifier) for (identifier,org) in originals.items() if org == ident]
			for num,mockindex in enumerate(mocks):
				xpos=i if len(mocks) == 1 else (i-1.0/3.0 + (2.0/3.0) * num / float(len(mocks)-1))
				print("%f %f" % (xpos, analysis.redundancyFactors[mockindex]))
		'''
	if args.write_setup_html_overview:
		args.write_setup_html_overview.write(htmlOverview(setup, analysis, title=args.setupdirectory))
		args.write_setup_html_overview.close()
	if args.obfuscation is None:
		if args.write_obfuscation_html_overview is not None:
			print("Error: Cannot write HTML overview '%s' without specifying an obfuscator" % args.write_obfuscation_html_overview[0], \
				file=sys.stderr)
		sys.exit(0)
	if args.write_obfuscation_html_overview is not None and len(args.write_obfuscation_html_overview) > len(args.obfuscation):
		print("Error: Cannot write HTML overview '%s' without specifying an obfuscator" %\
			args.write_obfuscation_html_overview[len(args.obfuscation)], file=sys.stderr)
	evl=createSimpleEvaluator(analysis)
	latex_lines=[]
	for directory,overview in zip(args.obfuscation, args.write_obfuscation_html_overview or (None for _ in args.obfuscation)):
		obf=readObfuscation(setup, directory+'/')
		#print("Obfuscation '%s' gets the score %f"% (directory, evl.calculateScore(obf)))
		if not args.suppress_final_score:
			print(evl.calculateScore(obf,args.max_score,analysis))
		if overview is not None:
			with open(overview, 'wt') as f:
				f.write(htmlOverview(setup, analysis, obf, title=args.setupdirectory+' '+directory))
