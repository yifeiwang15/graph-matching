class Graph:
    def __init__(self):
        self.map = dict()

    def fromDict(self, dictionary):
        self.map = dictionary

    def addNode(self, node, connection):
        '''
        :param node (Node):
        :param connection ([(int, int) ...]):
        '''
        for key in connection:
            map[key] = node
    def addEdge(self, key, value):
        '''
        :param key ((int, int)): the index of the added edge
        :param value (tensor): the attribute of the added edge
        '''
        idx1, idx2 = key
        map[(idx1, idx2)] = value
        map[(idx2, idx1)] = value