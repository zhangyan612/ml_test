from deep_learning.tests import World
from deep_learning.tests import Learner


def main():
    while True:
        # pick the right action
        agent = World.player
        max_act, max_val = Learner.max_Q(agent)
        print(max_act, max_val)
        (agent, a, r, agent_updated) = Learner.do_action(max_act)

        # updated q
        max_act, max_val = Learner.max_Q(agent_updated)
        print('updated action: %s, value:%s' %(max_act, max_val))

        # start game
        World.start_game()
        print('success score %s' % max_val)