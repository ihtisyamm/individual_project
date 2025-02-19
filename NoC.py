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
        self.exb = simpy.Store(env, capacity=100)    # extension buffer in a node
        self.next = None
        self.injecting = False
        """
        self.queue = [simpy.Store(env),
                      simpy.Store(env)]                 # ejection queue after packet is ejected
        self.link = [simpy.Resource(env, capacity=1),
                     simpy.Resource(env, capacity=1)]   # ejection link limit
        self.injecting = False                          # to check if injecting is in progress
        self.port = None

        self.reserve = False                            # implement livelock avoidance
        self.rQueue = simpy.Store(env)                  # reserve queue for livelock avoidance
        self.input = False                              # check if there is any input that will 
                                                        # want to be ejected or deflect
        self.output = False                             # checked if there is any output is being injected
        """

        self.env.process(self.process())

    def nextNode(self, node):
        self.next = node
        print(f"Node {self.id} connected to Node {node.id}")

    def injection(self, packet):
        self.injecting = True
        try:
            flits = packet.flitation()
            print(f"Node {self.id}: Packet {packet.id} injected at time {self.env.now} (dest: {packet.dest})")
            for flit in flits:
                if len(self.single.items) > 0:
                    print(f"Node {self.id}: Loop busy. Storing flit {flit.id} in EXB at time {self.env.now}")
                    yield self.exb.put(flit)
                else:
                    yield self.env.process(self.forward(flit))
                yield self.env.timeout(1)

            while len(self.exb.items) > 0:
                flit = yield self.exb.get()
                print(f"Node {self.id}: Draining flit {flit.id} from EXB at time {self.env.now}")
                yield self.env.process(self.forward(flit))
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
        print(f"Node {self.id}: forwarding flit {flit.id} to next Node {self.next.id} at time {self.env.now}")
        yield self.next.single.put(flit)
        yield self.env.timeout(1)

        # change the node also or should i?
        # does the node matter?
        # is the important thing is the (packet id, node id) or node or both?
        # should change the node, but how?
        # need to finish path asap
        # maybe just copy the path element, but how to know which node we're in right now?

    # determine where if the packet has arrived at the node destination
    def process(self):
        """
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
        """
        while True:
            flit = yield self.single.get()
            print(f"Node {self.id}: Processing flit {flit.id} at time {self.env.now}")

            if flit.dest == self.id:
                print(f"Node {self.id}: Ejecting flit {flit.id} at time {self.env.now}")
            else:
                yield self.env.process(self.forward(flit))

    # assigned a pre-determined path to each flit when it is injected
    def routing(self, flit):
        pass

# generate arbitrary packet size
# note: fixed to 3 to make it more simpler first
def create(env, node, pid=0):
    pid = 0 # again, packet id
    dest = (node.id + 1) % 16

    while True:
        packet = Packet(
            id=pid,
            src=node.id,
            dest=dest,
            size=5,
            time=env.now
        )
    
        yield env.process(node.injection(packet))
        pid += 1
        yield env.timeout(10)

def run(size=0):
    env = simpy.Environment()
    #node = Node(env, 1)
    #env.process(create(env, node))
    nodes = [Node(env, i) for i in range(size)]

    for i in range(size):
        nodes[i].nextNode(node=nodes[(i+1) % size])

    for i, node in enumerate(nodes):
        env.process(create(env, node=node, pid=i*1000))

    return env, nodes

if __name__ == "__main__":
    env, nodes = run(size=16)
    env.run(until=10)