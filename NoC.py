import simpy
import random

class Packet:
    def __init__(self, id, src, dest, size, time):
        self.id = id
        self.src = src
        self.dest = dest
        self.size = size
        self.timestamp = time
        self.flits = []

    def flitation(self):
        for i in range(self.size):
            type = "BODY"
            if i == 0:
                type = "HEAD"
            elif i == (self.size-1):
                type = "TAIL"
            
            self.flits.append(Flit(
                id=f"packet{self.id}_flit{i}",
                pid=self.id,
                src=self.src,
                dest=self.dest,
                time=self.timestamp
            ))
        return self.flits

class Flit:
    def __init__(self, id, pid, type, src, dest, time):
        self.id = id
        self.pid = pid # pid stands for packet id btw
        self.type = type # head | body | tail
        self.src = src
        self.dest = dest
        self.timestamp = time

class Node:
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.single = simpy.Store(env, capacity=1) # for single flit buffer
        self.shared = simpy.Store(env, capacity=100) # extension buffer in a node

    def injection(self, packet):
        pass

    def ejection(self):
        pass

    def send(self):
        yield 

    def process(self):
        pass

    def routing(self):
        pass
    
def create_packet(env, node):
    pid = 0 # again, packet id

    while True:
        packet = Packet(
            id=pid,
            src=node.id,
            dest=node.id,
            size=random.randint(3, 5),
            time=env.now
        )
    
        yield env.process(node.injection(packet))
        pid += 1
        yield env.timeout(random.randint(2, 5))

def run(duration=20):
    env = simpy.Environment()
    #node = Node(env)
    #env.process(create_packet(env, node))
    env.run(until=duration)

if __name__ == "__main__":
    run()