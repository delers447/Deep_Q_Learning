from Environment import EnvironmentClass
from GUI import GUI
import numpy as np
from matplotlib import pyplot as plt
import threading

##### PARAMETERS ####
STARTING_POS = [890.0, 140.0]
RACERS = ["Dan", "Kefe", "Chib", "Nick", "Rupo", "Stone", "Rob", "John", "Brian", "Keith"]
#RACERS_MODELS = ["Dan.model"]
JPEGFILENAME = "TrackRace(Gated).jpg"
N_GAMES = 250

#####################

def train(env, generation, parallel_letter):
	
	episodes = [[] for i in RACERS]
	average_scores = [[] for i in RACERS]
	eps_history = [[] for i in RACERS]
	scores = [[] for i in RACERS]
	avg_score = 0
	i = -1

	#model = env.save_model("dan", car_index=1)

	for i in range(len(RACERS)):
		env.load_model(F"{parallel_letter}_best_{generation-1}", car_index=i)

	for game in range(N_GAMES):
		print(F"===== NEW EPOSIDE, Generation {generation}, Game {game} ======")
		for i in range(len(RACERS)):
			score, epsilon, reward_hist = env.train(car_index=i)

			eps_history[i].append(epsilon)
			scores[i].append(score)
			avg_score = np.mean(scores[i][max(0, game-20):(game+1)])

			print(f"Episode {game}, Car: {RACERS[i]}, score: {score}, average_score: {round(avg_score, 2)}, Reward Hist: {reward_hist}")
			episodes[i].append(game)
			average_scores[i].append(avg_score)
	
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

	for episode, average_score in zip(episodes, average_scores):
		plt.plot(episode, average_score)
		#print(average_score)
	plt.xlabel("Average Score")
	plt.ylabel("Episodes")
	plt.title("Deep-Q Learning Average Scores")
	plt.show()


def race(env):
	cars = env.get_cars()
	print(env.TRACK_MAT.shape)

	#print(f"The legnth of the car history is {len(cars[0].front_bumper_history)}: {cars[0].front_bumper_history}")

	for car in cars:
		print(F"The disqualification_check reported: {env.disqualification_check()}.")
		while not env.disqualification_check():
			print(F"The car's position: {car.front_bumper_pos} with state: {env.get_input_distances()}.")
			state = env.get_input_distances()
			action = car.make_decision(state)
			#print(F"State: {state}")
			print(f"The legnth of the car history is {len(car.front_bumper_history)}")
			car.drive(action)
			print(f"The legnth of the car history is {len(car.front_bumper_history)}")
		
		directions = [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0),(0,0)]
		for car_x, car_y in car.front_bumper_history:
			for direction in directions:
				dx, dy = direction
				env.TRACK_MAT[int(car_x+dx), int(car_y+dy)] = 10
		env.track_displayer()
		print("+"*5, "CAR RESET", "+"*5)
		#env.get_cars()[0].displayCar_Info()

def main():
	env = EnvironmentClass(JPEGFILENAME, STARTING_POS, "East", RACERS)
	for i in range(1, 11,):
		train(env, i, 'a')
		env.purge_car_histories()
	env.reset_cars()
	#env.race()

if __name__ == "__main__":
	main()