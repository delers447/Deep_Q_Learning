from Environment import EnvironmentClass
from GUI import GUI
import numpy as np
from matplotlib import pyplot as plt
from threading import Thread
import time, gc

##### PARAMETERS ####
STARTING_POS = [890.0, 140.0]
RACERS = ["Dan", "Kefe", "Chib", "Nick", "Rupo", "Stone", "Longinus", "Maximillian", 
		  "Calista", "Benedicta", "Aquila", "Flora", "Zoroastres", "Servius", "Themistocles", "Achill"]

#RACERS_MODELS = ["Dan.model"]
JPEGFILENAME = "TrackRace(Gated).jpg"
N_GAMES = 5_000
 
#####################

class ThreadWithReturnValue(Thread):
	def __init__(self, group=None, target=None, name=None,
					args=(), kwargs=(), Verbose=None):
		Thread.__init__(self, group, target, name, args, kwargs)
		self._returnValue = None

	def run(self):
		if self._target is not None:
			self._returnValue = self._target(self._args)
			#print(self._returnValue)
	def join(self):
		Thread.join(self)
		return self._returnValue

def train(env, generation, parallel_letter):
	
	episodes = [[] for i in RACERS]
	average_scores = [[] for i in RACERS]
	eps_history = [[] for i in RACERS]
	scores = [[] for i in RACERS]
	threads = [[] for i in RACERS]
	avg_score = 0
	i = -1

	for i in range(len(RACERS)):
		env.load_model(F"{parallel_letter}_best_{generation-1}", car_index=i)

	for game in range(N_GAMES):
		threads = [[] for i in RACERS]
		start_time = time.time()
		top_average = 0
		top_racer = -1
		print(F"===== NEW EPISODE, Generation {generation}, Game {game} ======")
		for i in range(len(RACERS)):

			threads[i]= ThreadWithReturnValue(target=env.train, args = (i))
			threads[i].start()

		for i in range(len(RACERS)):
			outputs = threads[i].join()
			try:
				score, epsilon, reward_hist = outputs
			except:
				continue

			eps_history[i].append(epsilon)
			scores[i].append(score)
			avg_score = np.mean(scores[i][max(0, game-20):(game+1)])

			print(f"Episode {game}, Car: {RACERS[i]}, score: {score}, average_score: {round(avg_score, 2)}, Reward Hist: {[round(reward, 2) for reward in reward_hist]}")
			episodes[i].append(game)
			average_scores[i].append(avg_score)
			
			if avg_score > top_average:
				top_racer = i
				top_average = avg_score
		
		while len(threads)!=0:
			thread = threads.pop()
			del(thread)
		
		print(F"Generation {generation}, Game {game}: the best racer was {RACERS[top_racer]} with an average score of {top_average}.  Time Taken: {round(time.time() - start_time, 2)} Seconds ")
		print(F"===== End of EPISODE, Time Taken: {round(time.time() - start_time, 2)} Seconds ======")


	top_average = 0
	top_racer = 10

	for i in range(len(RACERS)):
		racers_average = np.mean(scores[i])
		print(F"{RACERS[i]}'s average was {racers_average}.")
		if racers_average > top_average:
			top_racer = i
			top_average = racers_average
	print(F"The best racer was {RACERS[top_racer]} with an average score of {top_average}.")

	env.save_model(f"{parallel_letter}_best_{generation}", car_index=top_racer)

	return top_average

	#for episode, average_score in zip(episodes, average_scores):
	#	plt.plot(episode, average_score)
	#	#print(average_score)
	#plt.xlabel("Average Score")
	#plt.ylabel("Episodes")
	#plt.title("Deep-Q Learning Average Scores")
	#plt.show()


def main():
	
	parallel_letter = 'g'
	log_filename = F"{parallel_letter}_generation_outcomes.csv"
	logfile = open(log_filename, 'w')
	generations = []
	top_averages  = []
	
	env = EnvironmentClass(JPEGFILENAME, STARTING_POS, "East", RACERS)
	env.create_random_models()
	env.save_model(f"{parallel_letter}_best_0", car_index=0)

	logfile.write("Generation, top_average")
	for i in range(1, 100):
		top_average = train(env, i, parallel_letter)
		logfile.write(F"{i}, {top_average}")
		generations.append(i)
		top_averages.append(top_average)
		#env.purge_car_histories()
		gc.collect()

	plt.plot(generations, top_averages)
	plt.xlabel("Generation")
	plt.ylabel("Average Score")
	plt.title("Deep-Q Learning, Learning Vs Generation")
	plt.show()
	print(top_averages)
	
	
	#The following can be used to test a specific model.
	#env = EnvironmentClass(JPEGFILENAME, STARTING_POS, "East", ["Dan"])
	#env.load_model(F"d_best_0", car_index=0)
	#env.reset_cars()
	#env.race()
	

if __name__ == "__main__":
	main()
