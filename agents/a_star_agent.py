import numpy as np


class Node:

    def __init__(self, parent=None, page=None):
        self.parent = parent
        self.page = page

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.page['title'] == other.page['title']


class AStarAgent(object):
    def __init__(self, data_handler, dim=512):
        self.data_handler = data_handler
        self.dim = dim

        self.initialized = False

        self.open_list = []
        self.closed_list = []

        self.current_node = None
        self.start_node = None
        self.end_node = None

    def setup(self, observation):
        self.start_node = Node(None, observation[0])
        self.start_node.g = self.start_node.h = self.start_node.f = 0
        self.end_node = Node(None, observation[1])
        self.end_node.g = self.end_node.h = self.end_node.f = 0
        self.current_node = self.start_node

    def act(self, observation, reward, done):

        if not self.initialized:
            self.setup(observation)
            self.initialized = True

        for new_page in observation[0]['refs']:
            new_node = Node(self.current_node, self.data_handler.get(new_page))

            if new_node in self.closed_list or new_node in self.open_list:
                continue

            # TODO handle possibility of discovering a new, shorter path to an existing node?
            new_node.g = self.current_node.g + 1
            new_node.h = 10*np.linalg.norm(self.end_node.page['title_embedding'] - new_node.page['title_embedding'])
            new_node.f = new_node.g + new_node.h

            self.open_list.append(new_node)

        current_node = self.open_list[0]
        current_index = 0
        for index, item in enumerate(self.open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        self.open_list.pop(current_index)
        self.closed_list.append(current_node)
        self.current_node = current_node

        if current_node.page['title'] in observation[0]['refs']:
            return {"type": "move", "direction": current_node.page['title_embedding']}
        else:
            return {"type": "return", "returning_state": current_node.page['title']}
