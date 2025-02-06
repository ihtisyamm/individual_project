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

    # segment packet into flits
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
                type=type,
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
        if self.type == "HEAD":
            self.cycle = 0

class Node:
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.single = simpy.Store(env, capacity=1)      # for single flit buffer
        self.shared = simpy.Store(env, capacity=100)    # extension buffer in a node
        self.queue = [simpy.Store(env),
                      simpy.Store(env)]                 # ejection queue after packet is ejected
        self.link = [simpy.Resource(env, capacity=1),
                     simpy.Resource(env, capacity=1)]   # ejection link limit
        self.injecting = False                          # to check if injecting is in progress
        self.port = None

        self.env.process(self.process())

    def injection(self, packet):
        self.injecting = True
        try:
            flits = packet.flitation()
            for flit in flits:
                yield self.env.process(self.send(flit))
        finally:
            self.injecting = False

    def ejection(self, flit):
        if len(self.queue[0].items) > 0:
            if self.queue[0].items[-1].pid == flit.pid:
                self.port = 0
            else:
                self.port = 1
        else:
            self.port = 1

        with self.link[self.port].request() as req:
            yield req
            yield self.queue[self.port].put(flit)
            yield self.env.timeout(1)

            if flit.type == "TAIL":
                print(f"Packet {flit.pid} ejected at time {self.env.now}")

    # forward to another node
    def send(self, flit):
        yield self.single.put(flit)

    # determine where if the packet has arrived at the node destination
    def process(self):
        while True:
            flit = yield self.single.get()
            yield self.env.process(self.ejection(flit))

            """
            
            if flit.pid == id:
                if len(self.link[0].items) == 1 and len(self.link[0].items) == 1:
                    yield self.env.process(self.send(flit))
                elif len(self.link[0].items) == 0 or len(self.link[1].items) == 0:
                    yield self.env.process(self.ejection(flit))
            """
    # assigned a pre-determined path to each flit when it is injected
    def routing(self, flit):
        pass

# generate arbitrary packet size
# note: fixed to 3 to make it more simpler first
def create(env, node):
    pid = 0 # again, packet id

    while True:
        packet = Packet(
            id=pid,
            src=node.id,
            dest=node.id,
            size=3,
            time=env.now
        )
    
        yield env.process(node.injection(packet))
        pid += 1
        yield env.timeout(1)

def run(duration=12):
    env = simpy.Environment()
    node = Node(env, 1)
    env.process(create(env, node))
    env.run(until=duration)

if __name__ == "__main__":
    run()