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

        self.reserve = False                            # implement livelock avoidance
        self.rQueue = simpy.Store(env)                  # reserve queue for livelock avoidance

        self.env.process(self.process())

    def injection(self, packet):
        self.injecting = True
        try:
            flits = packet.flitation()
            print(f"Node {self.id}: Packet {packet.id} injected at time {self.env.now}")
            for flit in flits:
                yield self.env.process(self.forward(flit))
                #if flit.type == "TAIL":
                 #   print(f"Packet {flit.pid} injected at time {self.env.now}")
                yield self.env.timeout(1)
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

        if self.reserve:
            yield self.timeout(1)
            self.reserve = False

        with self.link[self.port].request() as req:
            yield req
            yield self.queue[self.port].put(flit)

            if flit.type == "TAIL":
                print(f"Node {self.id}: Packet {flit.pid} ejected at time {self.env.now}")
            yield self.env.timeout(1)

    # forward to another node
    def forward(self, flit):
        yield self.single.put(flit)
        yield self.env.timeout(1)

    # determine where if the packet has arrived at the node destination
    def process(self):
        while True:
            flit = yield self.single.get()
            if flit.dest == self.id:

                eject = False
                for i in range(2):
                    if not self.link[i].users:
                        if flit.type == "HEAD":
                            eject = True
                            break
                        elif (len(self.queue[i].items) > 0 and
                            self.queue[i].items[-1].pid == flit.pid):
                            eject = True
                            break

                if eject:
                    yield self.env.process(self.ejection(flit))
                else:
                    if flit.type == "HEAD":
                        flit.cycle += 1
                    
                    if (flit.type == "HEAD" and
                        flit.cycle > 254):
                        self.reserve = True
                        yield self.rQueue.put(flit)
                    else:
                        yield self.env.process(self.forward(flit))
            else:
                yield self.env.process(self.forward(flit))

    # assigned a pre-determined path to each flit when it is injected
    def routing(self, flit):
        pass

# generate arbitrary packet size
# note: fixed to 3 to make it more simpler first
def create(env, node):
    pid = 0 # again, packet id
    dest = random.randint(0, 15)

    while True:
        packet = Packet(
            id=pid,
            src=node.id,
            dest=dest,
            size=3,
            time=env.now
        )
    
        yield env.process(node.injection(packet))
        pid += 1
        yield env.timeout(1)

def run(duration=100):
    env = simpy.Environment()
    #node = Node(env, 1)
    #env.process(create(env, node))
    nodes = []

    for i in range(16):
        nodes.append(Node(env, id=i))

    for node in nodes:
        env.process(create(env, node=node))
    env.run(until=duration)

if __name__ == "__main__":
    run()