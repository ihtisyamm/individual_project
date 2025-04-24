import simpy
import math
import time as time_module

# find every print statement and change it to something else

class Flit:
    def __init__(self, id, pid, type, src, dest, time, loopID=None):
        self.id = id
        self.packetID = pid
        self.type = type
        self.src = src
        self.dest = dest
        self.timestamp = time
        self.loopID = loopID
        self.circlingCount = 0
        self.currNode = src     # current node
        self.injectTime = None   # injection time: when the packet was injected
        self.ejectTime = None    # ejection time: when the packet ejected

class Packet:
    def __init__(self, id, src, dest, size, time):
        self.id = id
        self.src = src
        self.dest = dest
        self.size = size
        self.timestamp = time
        self.flits = []
        self.loopID = None

    # segment packet to flits
    def flitation(self):
        for i in range(self.size):
            type = "BODY"
            if i == 0:
                type = "HEAD"
            elif i == self.size-1:
                type = "TAIL"
            
            flit = Flit(
                id=f"packet{self.id}_flit{i}",
                pid=self.id,
                type=type,
                src=self.src,
                dest=self.dest,
                time=self.timestamp,
                loopID=self.loopID
            )
            self.flits.append(flit)
        return self.flits
    
class Loop:
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes
    
    def getNextNode(self, currNode):
        if currNode not in self.nodes:
            return None
        
        currNodeIndex = self.nodes.index(currNode)
        nextIndex = (currNodeIndex + 1)%(len(self.nodes) - 1)
        return self.nodes[nextIndex]
    
    def getPathLen(self, src, dest):
        srcIdx = self.nodes.index(src)  # idx stands for index for shorter form
        destIdx = self.nodes.index(dest)

        if destIdx < srcIdx:
            return len(self.nodes) - 1 - srcIdx + destIdx
        
        return destIdx - srcIdx
    
# to link predetermined path with nodes
class Link:
    def __init__(self, env, fromNode, toNode, loopID):
        self.env = env
        self.fromNode = fromNode
        self.toNode = toNode
        self.loopID = loopID
        self.buffer = simpy.Store(env, capacity=1)

# check this class again later
class Network:
    def __init__(self, env, loops):
        self.env = env
        self.loops = {loop.id: loop for loop in loops}
        self.links = {}     # (fromNode, toNode, loopID) -> Link

        self.createLink()

    def createLink(self):
        for loopID, loop in self.loops.items():
            for i in range(len(loop.nodes) - 1):
                fromNode = loop.nodes[i]
                toNode = loop.nodes[i+1]
                link = Link(self.env, fromNode, toNode, loopID)
                self.links[(fromNode, toNode, loopID)] = link

    def transmit(self, fromNode, toNode, loopID, flit):
        link = self.links.get((fromNode, toNode, loopID))
        if link:
            return link.buffer.put(flit)
        else:
            raise ValueError(f"No link from {fromNode} to {toNode} on loop {loopID}")
    
    # check this method again later
    def getIncomingLink(self, toNode, loopID):
        for key, link in self.links.items():
            if link.toNode == toNode and link.loopID == loopID:
                return link
        return None
    
class Node:
    def __init__(self, env, nodeID, network, EXBSize=5, ejectLinks=2):
        self.env = env
        self.nodeID = nodeID
        self.network = network

        # to dfetermine loops for the nodes
        self.loops = self.getLoopsForNode()

        # create buffers and resources
        self.loopBuffers = {loopID: simpy.Store(env, capacity=1) for loopID in self.loops} # check this again later. i mean what's this
        self.exb = simpy.Store(env, capacity=EXBSize)
        self.ejectionLinks = [simpy.Resource(env, capacity=1) for _ in range(ejectLinks)]
        self.ejectionQueues = [simpy.Store(env) for _ in range(ejectLinks)]

        self.routingTable = {}  # {dest: bestLoopID}
        self.exbAttachedToLoop = {loopID: False for loopID in self.loops}   # check this line again later
        self.reservedEjection = None

        # for statistics later on
        self.injectedPackets = 0
        self.ejectedPackets = 0
        self.packetLatencies = []

        for loopID in self.loops:
            link = network.getIncomingLink(self.nodeID, loopID)
            if link:
                self.env.process(self.monitorLink(link))
    
    def getLoopsForNode(self):
        loopsNode = []
        for loopID, loop in self.network.loops.items():
            if self.nodeID in loop.nodes:
                loopsNode.append(loopID)
        return loopsNode
    
    # for debugging log if it enamel. LOL! *if it enabled
    # def log(self, message):
    #     if self.debug:
    #         print(f"Time {self.env.now}: Node {self.node_id} - {message}")

    def injection(self, packet):
        self.injectedPackets += 1
        yield self.env.timeout(0)   # check this line again later

        bestLoopID = self.findBestLoop(packet.dest)

        if bestLoopID is None:
            print(f"No loop available for destination {packet.dest}, packet {packet.id} dropped")
            return
        
        packet.loopID = bestLoopID

        flits = packet.flitation()
        print(f"\n=== INJECTING packet {packet.id} to destination {packet.dest} via loop {bestLoopID} ===")

        # check this logic. not sure why EXB is attached????
        # it said if it long packet it will be attached to EXB. like why??
        if len(flits) > 1:
            if not self.exbAttachedToLoop[bestLoopID]:
                
                if len(self.exb.items) >= self.exb.capacity:
                    print(f"EXB FULL: Cannot inject packet {packet.id}, waiting")
                    # wait for EXB to becaome avaialbel
                    yield self.env.timeout(1)
                    # retry injection
                    yield self.env.process(self.injection(packet))
                    return
                
                self.exbAttachedToLoop[bestLoopID] = True
                print(f"Attaching EXB to loop {bestLoopID} for packet {packet.id}")
                print(f"DEBUG-EXB: Loop {bestLoopID} for packet {packet.id}, EXB items: {len(self.exb.items)}/{self.exb.capacity}")
                print(f"TRACE: Packet {packet.id} from {packet.src} to {packet.dest}, size {len(flits)}")


        for flit in flits:
            # checking if the loop is available
            if len(self.loopBuffers[bestLoopID].items) == 0:
                print(f"Injecting flit {flit.id} into loop {bestLoopID}")
                print(f"TRACE: Packet {packet.id} from {packet.src} to {packet.dest}, size {len(flits)}")
                print(f"TRACE: Flit {flit.id} entering system at {self.env.now}, EXB={self.exbAttachedToLoop[bestLoopID]}")


                # update the flit's current node
                flit.currNode = self.nodeID
                flit.injectTime = self.env.now

                flit.injectTime = self.env.now
                print(f"DEBUG-INJECT: Flit {flit.id} injected at time {flit.injectTime}, env time {self.env.now}")
                print(f"TRACE: Packet {packet.id} from {packet.src} to {packet.dest}, size {len(flits)}")
                print(f"TRACE: Flit {flit.id} entering system at {self.env.now}, EXB={self.exbAttachedToLoop[bestLoopID]}")


                
                yield self.loopBuffers[bestLoopID].put(flit)
                self.env.process(self.processFlit(bestLoopID))
            else:
                print(f"STORING IN EXB: Loop {bestLoopID} busy - flit {flit.id} goes to EXB")
                print(f"DEBUG-EXB: Storing flit {flit.id} in EXB, current size: {len(self.exb.items)}/{self.exb.capacity}")
                print(f"TRACE: Packet {packet.id} from {packet.src} to {packet.dest}, size {len(flits)}")
                print(f"TRACE: Flit {flit.id} entering system at {self.env.now}, EXB={self.exbAttachedToLoop[bestLoopID]}")
                if flit.injectTime is None:
                    flit.injectTime = self.env.now

                yield self.exb.put(flit)

            yield self.env.timeout(1)

        print(f"=== COMPLETED INJECTION of packet {packet.id} ===\n")

        if self.exbAttachedToLoop[bestLoopID] and len(self.exb.items) > 0:
            yield self.env.process(self.drainEXB(bestLoopID))

    def drainEXB(self, loopID):
        print(f"Attempting to drain EXB for loop {loopID}")

        while len(self.exb.items) > 0 and self.exbAttachedToLoop[loopID]:
            if len(self.loopBuffers[loopID].items) == 0:
                flit = yield self.exb.get()
                print(f"DRAINING FROM EXB: flit {flit.id} to loop {loopID}")
                print(f"DEBUG-EXB: Draining EXB for loop {loopID}, items: {len(self.exb.items)}/{self.exb.capacity}")

                flit.currNode = self.nodeID
                # Don't reset inject time if it's already set
                # flit.injectTime = self.env.now  <- Remove or comment out this line

                yield self.loopBuffers[loopID].put(flit)
                self.env.process(self.processFlit(loopID))

                # if EXB is empty, detach from loop
                # check this logic later
                if len(self.exb.items) == 0:
                    self.exbAttachedToLoop[loopID] = False
                    print(f"Detaching EXB from loop {loopID} - EXB is empty")
            else:
                print(f"Cannot drain EXB - loop {loopID} is busy")
                break

            yield self.env.timeout(1)

    def monitorLink(self, link):
        while True:
            flit = yield link.buffer.get()
            flit.currNode = self.nodeID
            print(f"Received flit {flit.id} from loop {link.loopID}")

            if len(self.loopBuffers[link.loopID].items) == 0:
                yield self.loopBuffers[link.loopID].put(flit)

                self.env.process(self.processFlit(link.loopID))
            else:
                print(f"ERROR: Loop {link.loopID} buffer occupied when receiving flit {flit.id}")

    def processFlit(self, loopID):
        flit = yield self.loopBuffers[loopID].get()

        print(f"Processing flit {flit.id} on loop {loopID}")

        # checking correct destination
        if flit.dest == self.nodeID:
            success = yield self.env.process(self.eject(flit))
            if not success:
                # failed if for example ejection links busy
                yield self.env.process(self.forward(flit))
        else:
            # not for this node, forward
            yield self.env.process(self.forward(flit))

    def forward(self, flit):
        loopID = flit.loopID
        loop = self.network.loops[loopID]

        nextNode = loop.getNextNode(self.nodeID)
        if nextNode is None:
            print(f"ERROR: Cannot find next node for flit {flit.id} on loop {loopID}")
            return
        print(f"Forwarding flit {flit.id} on loop {loopID} to node {nextNode}")

        # Add a delay of 1 cycle here before the flit is transmitted
        yield self.env.timeout(1)  # This enforces that each hop takes at least 1 cycle

        # check again what is transmit
        yield self.network.transmit(self.nodeID, nextNode, loopID, flit)

    def eject(self, flit):
        if self.reservedEjection is not None and flit.circlingCount >= 254:
            ejectionLink = self.ejectionLinks[self.reservedEjection]

            with ejectionLink.request() as req:
                yield req

                yield self.ejectionQueues[self.reservedEjection].put(flit)
                print(f"Ejecting flit {flit.id} via reserved ejection link {self.reservedEjection}")

                # this is for statistics or results
                if flit.type == "HEAD":
                    flit.ejectTime = self.env.now
                    latency = flit.ejectTime - flit.injectTime
                    print(f"DEBUG-EJECT: Flit {flit.id} ejected at time {flit.ejectTime}, env time {self.env.now}")
                    print(f"DEBUG-LATENCY: Flit {flit.id} latency calculation: {flit.ejectTime} - {flit.injectTime} = {latency}")

                if flit.type == "TAIL":
                    print(f"TRACE: Complete packet {flit.packetID} ejected, head latency={self.packetLatencies[-1]}")
                    print(f"TRACE: Full packet time: {self.env.now - flit.timestamp}")
                    print(f"Completed ejection of packet {flit.pid}")
                    self.ejectedPackets += 1
                    self.reservedEjection = None

                yield self.env.timeout(1)
                return True
            
        for linkIdx, ejectionLink in enumerate(self.ejectionLinks):
            if ejectionLink.count == 0:
                with ejectionLink.request() as req:
                    yield req

                    yield self.ejectionQueues[linkIdx].put(flit)
                    print(f"Ejecting flit {flit.id} via ejection link {linkIdx}")

                    if flit.type == "HEAD":
                        flit.ejectTime = self.env.now
                        latency = flit.ejectTime - flit.injectTime
                        self.packetLatencies.append(latency)
                        print(f"LATENCY: Flit {flit.id} injected at {flit.injectTime}, ejected at {flit.ejectTime}, latency={latency}")

                    if flit.type == "TAIL":
                        print(f"Completed ejection of packet {flit.packetID}")
                        self.ejectedPackets += 1

                    yield self.env.timeout(1)
                    return True
                
        if flit.type == "HEAD":
            flit.circlingCount += 1
            print(f"No ejection links available - flit {flit.id} circling (count: {flit.circlingCount})")

            if flit.circlingCOunt >= 254 and self.reservedEjection is None:
                self.reservedEjection = 0

        return False
    
    def findBestLoop(self, dest):
        bestLoopID = None
        minHops = float('inf')

        for loopID in self.loops:
            loop = self.network.loops[loopID]
            # Check if both source and destination are in this loop
            if self.nodeID in loop.nodes and dest in loop.nodes:
                pathLen = loop.getPathLen(self.nodeID, dest)
                if pathLen < minHops:
                    minHops = pathLen
                    bestLoopID = loopID

        if bestLoopID is not None:
            self.routingTable[dest] = bestLoopID

        return bestLoopID
    
# check this class again later
class RouterlessNoC:
    def __init__(self, size):
        self.size = size
        self.env = simpy.Environment()
        
        self.loops = self.generateLoops()
        self.network = Network(self.env, self.loops)
        self.nodes = self.createNodes()

        # stats
        self.packetCounter = 0
        self.stats = {
            'injected': 0,
            'ejected': 0,
            'latencies': []
        }

        # check this seed stuff. dont think id ever need it
        self.seed = int(time_module.time())

    def generateLoops(self):
        loops = []
        size = self.size

        if size == 2:
            loops.append(Loop(0, [0, 1, 3, 2, 0])) # clockwise
            loops.append(Loop(1, [0, 2, 3, 1, 0])) # counter-clockwise
        elif size == 4:
            # layer 1 (inner loops)
            inner1 = [5, 6, 10, 9, 5] # clockwise
            inner2 = [5, 9, 10, 6, 5] # counter-clockwise

            # layer 2 (outer loops)
            # goup A
            outer1 = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 0]
            
            # group B
            outer2 = [0, 4, 8, 12, 13, 9, 5, 1, 0]
            outer3 = [0, 4, 8, 12, 13, 14, 10, 6, 2, 1, 0]

            # group C
            outer4 = [3, 2, 6, 10, 14, 15, 11, 7, 3]
            outer5 = [3, 2, 1, 5, 9, 13, 14, 15, 11, 7, 3]

            # group D
            outer6 = [0, 1, 2, 3, 0]
            outer7 = [4, 5, 6, 7, 4]
            outer8 = [8, 9, 10, 11, 8]
            outer9 = [12, 13, 14, 15, 12]

            loops.append(Loop(0, inner1))
            loops.append(Loop(1, inner2))
            loops.append(Loop(2, outer1))
            loops.append(Loop(3, outer2))
            loops.append(Loop(4, outer3))
            loops.append(Loop(5, outer4))
            loops.append(Loop(6, outer5))
            loops.append(Loop(7, outer6))
            loops.append(Loop(8, outer7))
            loops.append(Loop(9, outer8))
            loops.append(Loop(10, outer9))
        else:
            # check this rlrec later
            # implement RLrec algorithm if larger sizes
            perimeterNodes = []

            # top row left to right
            for i in range(size):
                perimeterNodes.append(i)
            
            # right column top to bottom
            for i in range(1, size):
                perimeterNodes.append(i*size + (size-1))

            # bottom row right to left
            for i in range(size-2, -1, -1):
                perimeterNodes.append((size-1)*size + i)

            # left column bottom to top
            for i in range(size-2, 0, -1):
                perimeterNodes.append(i*size)

            # check this again why need to back to start
            # back to start 
            perimeterNodes.append(0)

            loops.append(Loop(0, perimeterNodes))

            # horizontal rows
            for i in range(size):
                rowNodes = [i*size + j for j in range(size)]
                rowNodes.append(rowNodes[0]) # back to start
                loops.append(Loop(i+1, rowNodes))

            # vertical columns
            for i in range(size):
                columnNodes = [j*size + i for j in range(size)]
                columnNodes.append(columnNodes[0]) # back to start
                loops.append(Loop(size+i+1, columnNodes))

            print("Generated Loops: ")
            for loop in loops:
                print(loop)

        return loops
        
    def createNodes(self):
        nodes = {}
        size = self.size
        totalNodes = size * size

        for nodeID in range(totalNodes):
            nodes[nodeID] = Node(env=self.env,
                                 nodeID=nodeID,
                                 network=self.network)
        return nodes
    
    def generatePacket(self, src, dest=None, size=3, time=None):
        if time is None:
            time = self.env.now
        
        # check this again later
        # if dest is not specified, choose a random destination
        if dest is None:
            size = self.size
            totalNodes = size * size
            dest = self.randomInt(0, totalNodes - 1)

        packetID = self.packetCounter
        self.packetCounter += 1

        return Packet(id=packetID,
                      src=src,
                      dest=dest,
                      size=size,
                      time=time)
    
    # check this again later
    def inject(self, src, dest=None, size=3, time=None):
        packet = self.generatePacket(src, dest, size, time)
        self.stats['injected'] += 1

        # check this again later
        # schedule the packet injection
        if time is not None and time > self.env.now:
            delay = time - self.env.now
            self.env.process(self.delayedInjection(src, packet, delay))
        else:
            # check this again later
            self.env.process(self.nodes[src].injection(packet))

        return packet
    
    def delayedInjection(self, src, packet, delay):
        yield self.env.timeout(delay)
        yield self.env.process(self.nodes[src].injection(packet))

    # check this function again later
    def run(self, until=100):
        self.stats = {'injected': 0,
                      'ejected': 0,
                      'latencies': []}
        
        self.env.run(until=until)

        # gather the stats
        for nodeID, node in self.nodes.items():
            self.stats['injected'] += node.injectedPackets
            self.stats['ejected'] += node.ejectedPackets
            self.stats['latencies'].extend(node.packetLatencies) # check this. like what does it do actually

        # calculate latency
        if self.stats['latencies']:
            self.stats['avgLatency'] = sum(self.stats['latencies']) / len(self.stats['latencies'])
        else:
            self.stats['avgLatency'] = 0

        #print(f"DEBUG-STATS: Total packet latencies collected: {len(self.stats['latencies'])}")
        #print(f"DEBUG-STATS: Sample of latencies: {self.stats['latencies'][:10]}")

        return self.stats
    
    # check these functions later
    def randomInt(self, low, high):
        # check this random algorithm again later
        self.seed = (1103515245 * self.seed + 12345) & 0x7fffffff
        return low + (self.seed % (high - low + 1))
    
    def randomFloat(self):
        self.seed = (1103515245 * self.seed + 12345) & 0x7fffffff
        return self.seed / 0x7fffffff
    
    def exponential(self, rate):
        u = self.randomFloat()
        return -math.log(1.0 - u) / rate
    
    def generateTraffic(self, injectionRate, packetSize=3, duration=1000):
        size = self.size
        totalNodes = size * size

        # check this algorithm later
        # he used poisson process whatsoever
        # dont think id ever need it
        # original line 656
        for src in range(totalNodes):
            nextTime = 0
            while nextTime < duration:
                interArrival = self.exponential(injectionRate)
                nextTime += interArrival

                if nextTime < duration:
                    dest = self.randomInt(0, totalNodes - 1)
                    while dest == src:
                        dest = self.randomInt(0, totalNodes - 1)
                    self.inject(src, dest, packetSize, nextTime)

# simulation run
def runNoC(totalPackets=20):
    noc = RouterlessNoC(size=4)

    totalNodes = noc.size * noc.size

    # check this again later. have no idea what this
    for i in range(totalPackets):
        src = i % totalNodes
        dest = (src+1) % totalNodes
        injectionTime = i*2
        packetSize = 3 + (i%3)

        noc.inject(src, dest, packetSize, injectionTime)

    duration = (totalPackets*2) + 100
    stats = noc.run(until=duration)

    # Print statistics
    print("\nManual Traffic Simulation Statistics:")
    print(f"Injected packets: {stats['injected']}")
    print(f"Ejected packets: {stats['ejected']}")
    if stats['latencies']:
        print(f"Average packet latency: {stats['avgLatency']:.2f} cycles")
        print(f"Min latency: {min(stats['latencies'])} cycles")
        print(f"Max latency: {max(stats['latencies'])} cycles")
        
        # Add a histogram of latencies
        latency_buckets = {}
        for lat in stats['latencies']:
            if lat in latency_buckets:
                latency_buckets[lat] += 1
            else:
                latency_buckets[lat] = 1
        
        print("\nLatency distribution:")
        for lat in sorted(latency_buckets.keys()):
            print(f"  {lat} cycles: {latency_buckets[lat]} packets")

# check if exb working as intended
def test_exb_contention():
    noc = RouterlessNoC(size=2)
    
    # Force two packets from same source close together to create contention
    noc.inject(src=0, dest=1, size=5, time=0)  # Larger packet
    noc.inject(src=0, dest=3, size=3, time=1)  # Second packet before first one is fully injected
    
    stats = noc.run(until=20)
    print("\nEXB Contention Test Results:")
    print(f"Injected packets: {stats['injected']}")
    print(f"Ejected packets: {stats['ejected']}")

# Function to run a performance evaluation
def evaluate_performance(network_sizes=[2, 4], injection_rates=[0.01, 0.05, 0.1]):
    results = {}
    
    for size in network_sizes:
        results[size] = {'latency': {}, 'throughput': {}}
        
        for rate in injection_rates:
            # Run simulation with this injection rate
            noc = RouterlessNoC(size=size)
            noc.generate_random_traffic(injection_rate=rate, duration=1000)
            stats = noc.run(until=1100)  # Run for slightly longer to finish processing
            
            # Record results
            results[size]['latency'][rate] = stats['avgLatency'] if stats['latencies'] else 0
            results[size]['throughput'][rate] = stats['ejected'] / 1000  # Packets per cycle
    
    # Print results instead of plotting
    print("\nPerformance Evaluation Results:")
    print("==============================")
    
    for size in network_sizes:
        print(f"\n{size}x{size} Network:")
        print("  Injection Rate | Average Latency | Throughput")
        print("  --------------|----------------|------------")
        
        for rate in injection_rates:
            latency = results[size]['latency'][rate]
            throughput = results[size]['throughput'][rate]
            print(f"  {rate:.3f}        | {latency:.2f} cycles    | {throughput:.4f} packets/cycle")
    
    return results

if __name__ == "__main__":
    # Run a simple simulation
    #run_routerless_noc_simulation()
    runNoC(100)
    
    # Uncomment to run a performance evaluation
    #evaluate_performance()

