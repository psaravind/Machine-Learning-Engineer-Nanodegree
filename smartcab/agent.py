import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
from collections import namedtuple

Q_Tuple = namedtuple('Q_Tuple', [
	'light', 
	'oncoming', 
	'left',
	#'right',
	'next_waypoint', 
	#'deadline',
	'action'])

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		# TODO: Initialize any additional variables here
		self.Q = {} # Q table
		self.state = {}
		self.success = 0
		self.failure = 0
		self.count = 0
		self.penalty = 0
		self.net_reward = 0
		#self.deadline = 0

	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required	

	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)

		# TODO: Update state
		self.state['light'] = inputs['light']
		self.state['oncoming'] = inputs['oncoming']
		self.state['left'] = inputs['left']
		#self.state['right'] = inputs['right']
		self.state['next_waypoint'] = self.next_waypoint
		#self.state['deadline'] = deadline

		# TODO: Select action according to your policy
		action = self.selectAction()

		# Execute action and get reward
		reward = self.env.act(self, action)

		# collect statistics
		self.count += 1
		if reward >= 10:
			self.success += 1
		if deadline == 0:
			self.failure += 1
		if reward < 0:
			self.penalty += 1
		self.net_reward += reward
		self.deadline = deadline

		# TODO: Learn policy based on state, action, reward
		self.updateQ(action, reward)
		#print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

	def selectAction(self):
		if self.epsilon > random.randint(1, 2000):
			self.epsilon -= 1
			return random.choice(self.env.valid_actions)
		return self.bestAction()

	def bestAction(self):
		value = None
		best_action = None
		for action in self.env.valid_actions:
			if value == None or self.getQ(action) > value:
				best_action = action
				value = self.getQ(action)
			# resolve ties by random selection
			elif self.getQ(action) == value:	
				if random.choice((True, False)) == True:
					best_action = action
					value = self.getQ(action)
				
		if best_action != None:
			return best_action
		return random.choice(self.env.valid_actions)

	def getQ(self, action):
		key = Q_Tuple(
			light=self.state['light'], 
			oncoming=self.state['oncoming'], 
			left=self.state['left'], 
			#right=self.state['right'],
			next_waypoint=self.state['next_waypoint'], 
			#deadline=self.state['deadline'],
			action=action)
		if key in self.Q.keys():
			return self.Q[key]
		return 0

	def updateQ(self, action, reward):
		key = Q_Tuple(
			light=self.state['light'], 
			oncoming=self.state['oncoming'], 
			left=self.state['left'], 
			#right=self.state['right'],
			next_waypoint=self.state['next_waypoint'],
			#deadline=self.state['deadline'],
			action=action)
		old_value = 0
		if key in self.Q.keys():
			old_value = self.Q[key]

		inputs = self.env.sense(self)
		next_state = {}
		next_state['light'] = inputs['light']
		next_state['oncoming'] = inputs['oncoming']
		next_state['left'] = inputs['left']
		#next_state['right'] = inputs['right']
		next_state['next_waypoint'] = self.next_waypoint
		#next_state['deadline'] = self.state['deadline']
		
		# from https://en.wikipedia.org/wiki/Q-learning#Learning_rate:
		# Q(s, a) <- old_value + learning_rate * ( learned value - old value)
		# learned value = reward + discount_factor * estimate of optimal future value
		# Q(s, a) <- old_value + learning_rate * ( reward + discount_factor * estimate of optimal future value) - old value)
		# Q(s, a) <- Q(s, a) + learning_rate * ( reward + discount_factor * estimate of optimal future value) - Q(s, a))
		
		self.Q[key] = old_value + (self.alpha * (reward + (self.gamma * self.getMaxQ(next_state)) - old_value))

	def getMaxQ(self, state):
		max = None

		for action in self.env.valid_actions:
			key = Q_Tuple(
				light=state['light'], 
				oncoming=state['oncoming'], 
				left=state['left'], 
				#right=state['right'], 
				next_waypoint=state['next_waypoint'], 
				#deadline=state['deadline'],
				action=action)
			if key in self.Q.keys():
				if max == None or max < self.Q[key]:
					max = self.Q[key]

		if max == None:
			return 0
		return max

def run():
	"""Run the agent for a finite number of trials."""
	
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent

	#set alpha, gamma & epsilon
	a.alpha = 0.2 #learning rate
	a.gamma = 0.15 #discount factor
	a.epsilon = 100 #exploration rate

	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
	# NOTE: You can set enforce_deadline=False while debugging to allow longer trials

	# Now simulate it
	sim = Simulator(e, update_delay=.00025, display=False)  # create simulator (uses pygame when display=True, if available)
	# NOTE: To speed up simulation, reduce update_delay and/or set display=False

	sim.run(n_trials=90)

	print "count:", a.count, "success:", a.success, " failure:", a.failure, "penalty:", a.penalty, "net_reward", a.net_reward, "time remaining:", a.deadline, "size:", len(a.Q.keys())
	
	sim.run(n_trials=10)

	print "count:", a.count, "success:", a.success, " failure:", a.failure, "penalty:", a.penalty, "net_reward", a.net_reward, "time remaining:", a.deadline, "size:", len(a.Q.keys())
	
	print a.Q
	exit()
	
	# grid search to find optimal value for alpha, gamma and epsilon
	for alpha in np.arange(0, 1, 0.05): # generate learning rate
		for gamma in np.arange(0, 1, 0.05): # discount factor
			for epsilon in range(100, 400, 100): # exploration rate
    # Set up environment and agent
				e = Environment()  # create environment (also adds some dummy traffic)
				a = e.create_agent(LearningAgent)  # create agent

				#set alpha, gamma & epsilon
				a.alpha = alpha
				a.gamma = gamma
				a.epsilon = epsilon

				e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
	# NOTE: You can set enforce_deadline=False while debugging to allow longer trials

	# Now simulate it
				sim = Simulator(e, update_delay=.00025, display=False)  # create simulator (uses pygame when display=True, if available)
	# NOTE: To speed up simulation, reduce update_delay and/or set display=False

				sim.run(n_trials=100)  # run for a specified number of trials
	# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line	
				print "alpha:", alpha, "gamma:", gamma, "epsilon:", epsilon, "count:", a.count, "success:", a.success, " failure:", a.failure, "penalty:", a.penalty, "net_reward", a.net_reward, "time remaining:", a.deadline, "size:", len(a.Q.keys())       

	exit()

if __name__ == '__main__':
	run()
