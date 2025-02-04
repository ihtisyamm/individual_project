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
        self.pid = pid      # pid stands for packet id btw
        self.type = type    # head | body | tail
        self.src = src
        self.dest = dest
        self.timestamp = time

class Node:
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.single = simpy.Store(env, capacity=1)      # for single flit buffer
        self.shared = simpy.Store(env, capacity=100)    # extension buffer in a node
        self.queue = simpy.Store(env)                   # ejection queue after packet is ejected
        self.link = simpy.Resource(env, capacity=2)     # ejection link limit
        
        self.env.process(self.process())

    def injection(self, packet):
        flits = packet.flitation()
        for flit in flits:
            yield self.env.process(self.send(flit))

    def ejection(self, packet):
        with self.link.request() as req:
            yield req
            yield self.queue.put(flit)
            yield self.env.timeout(1)

            if flit.type == "TAIL":
                print(f"Packet {flit.pid} ejected at time {self.env.now}")

    # forward to another node
    def send(self, flit):
        yield self.single.put(flit)

    # determine where if the packet has arrived at the node destination
    def process(self):
        if node:
            yield self.env.process(self.ejection(packet))
        else:
            yield self.send(flit)

    # assigned a pre-determined path to each flit when it is injected
    def routing(self, flit):
        pass

# generate arbitrary packet size
def create(env, node):
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