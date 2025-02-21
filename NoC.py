import simpy

class Path:
    def __init__(self, source, destination, nodes):
        self.source = source
        self.destination = destination
        self.path = nodes
        self.hops = len(nodes) - 1

class PathTable:
    def __init__(self, size):
        self.paths = {}
        self.loops = []
        self.size = size # network size

    def addLoop(self, loop):
        self.loops.append(loop)

        size = len(loop)
        for i in range(size):
            for j in range(size):
                src = loop[i]
                dest = loop[j]

                # clockwise path
                if j > i:
                    path = loop[i:j+1]
                else:
                    path = loop[i:] + loop[:j+1]

                key = (src, dest)

                if key not in self.paths or len(path) < len(self.paths[key].path):
                    self.paths[key] = Path(src, dest, path)

    def getPath(self, source, destination):
        return self.paths.get((source, destination))
    

class Packet:
    def __init__(self, id, src, dest, size, time):
        self.id = id
        self.src = src
        self.dest = dest
        self.size = size
        self.timestamp = time
        self.flits = []
        self.path = None

    # segment packet into flits
    def flitation(self):
        for i in range(self.size):
            type = "BODY"
            if i == 0:
                type = "HEAD"
            elif i == (self.size-1):
                type = "TAIL"
            
            flit = Flit(
                id=f"packet{self.id}_flit{i}",
                pid=self.id,
                type=type,
                src=self.src,
                dest=self.dest,
                time=self.timestamp
            )
            flit.path = self.path
            self.flits.append(flit)
        return self.flits

class Flit:
    def __init__(self, id, pid, type, src, dest, time):
        self.id = id
        self.pid = pid      # pid stands for packet id btw
        self.type = type    # head | body | tail
        self.src = src
        self.dest = dest
        self.path = None
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
        self.queue = [simpy.Store(env),
                      simpy.Store(env)]                 # ejection queue after packet is ejected
        self.link = [simpy.Resource(env, capacity=1),
                     simpy.Resource(env, capacity=1)]   # ejection link limit
        """
        self.injecting = False                          # to check if injecting is in progress
        self.port = None

        self.reserve = False                            # implement livelock avoidance
        self.rQueue = simpy.Store(env)                  # reserve queue for livelock avoidance
        self.input = False                              # check if there is any input that will 
                                                        # want to be ejected or deflect
        self.output = False                             # checked if there is any output is being injected
        """

        self.env.process(self.activity())
        

    def nextNode(self, node):
        self.next = node
        print(f"Node {self.id} connected to Node {node.id} at time {self.env.now}")

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
        """
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
        """
        for i, link in enumerate(self.link):
            if not link.users:
                if flit.type == "HEAD" or (
                    len(self.queue) > 0 and
                    self.queue[i].items[-1].pid == flit.pid
                ):
                    with link.request() as req:
                        yield req
                        yield self.queue[i].put(flit)
                        print(f"Node {self.id}: Flit {flit.id} ejected on link {i} at time {self.env.now}")
                        if flit.type == "TAIL":
                            print(f"Node {self.id}: Packet {flit.id} ejected at time {self.env.now}")
                        yield self.env.timeout(1)
                        return True
        return False

    # forward to another node
    def forward(self, flit):
        """
        if hasattr(flit, 'path'):
            try:
                current_idx = flit.path.path.index(self.id)
                if current_idx < len(flit.path.path) - 1:
                    next_node_id = flit.path.path[current_idx + 1]
                    print(f"Node {self.id}: forwarding flit {flit.id} to next Node {next_node_id} (path-based) at time {self.env.now}")
            except ValueError:
                pass

        print(f"Node {self.id}: forwarding flit {flit.id} to next Node {self.next.id} at time {self.env.now}")
        yield self.next.single.put(flit)
        yield self.env.timeout(1)
        """
        if hasattr(flit, 'path') and flit.path:
            try:
                current_idx = flit.path.path.index(self.id)
                if current_idx < len(flit.path.path) - 1:
                    next_node_id = flit.path.path[current_idx + 1]
                    next_node = self.node_lookup[next_node_id]
                    print(f"Node {self.id}: forwarding flit {flit.id} to next Node {next_node_id} at time {self.env.now}")
                    yield next_node.single.put(flit)
                    yield self.env.timeout(1)
                    return
            except ValueError:
                pass
        
        # Fallback to default behavior
        print(f"Node {self.id}: forwarding flit {flit.id} to next Node {self.next.id} at time {self.env.now}")
        yield self.next.single.put(flit)
        yield self.env.timeout(1)

    # determine where if the packet has arrived at the node destination
    def activity(self):
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
                checkEject = yield self.env.process(self.ejection(flit=flit))
                if not checkEject:
                    print(f"Node {self.id}: Failed to eject flit {flit.id}. Forward to another loop at time {self.env.now}")
                    yield self.env.process(self.forward(flit))
            else:
                yield self.env.process(self.forward(flit))

    # assigned a pre-determined path to each flit when it is injected
    def routing(self, flit):
        pass

# generate arbitrary packet size
# note: fixed to 3 to make it more simpler first
def create(env, nodes, pathTable):
    pid = 0 # again, packet id

    for i, node in enumerate(nodes):

        yield env.timeout(i*2)
        dest = (node.id + 1) % len(nodes)

        packet = Packet(
            id=pid,
            src=node.id,
            dest=dest,
            size=5,
            time=env.now # inject the packet in interval
        )

        path = pathTable.getPath(node.id, dest)
        if path:
            packet.path = path
    
        yield env.process(node.injection(packet))
        pid += 1

def run(size=0):
    env = simpy.Environment()
    nodes = [Node(env, i) for i in range(size)]

    node_lookup = {node.id: node for node in nodes}
    for node in nodes:
        node.node_lookup = node_lookup  # Add this to each node

    pathTable = PathTable(size=size)

    for i in range(size):
        next = nodes[(i+1) % size] # assign to next node
        nodes[i].nextNode(next)

        pathTable.addLoop([j % size for j in range(i, i + size)])

    env.process(create(env, nodes=nodes, pathTable=pathTable))
    return env, nodes

if __name__ == "__main__":
    env, nodes = run(size=16)
    env.run(until=5)