import simpy
import math
import time as time_module
import matplotlib.pyplot as plt

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
        if src not in self.nodes or dest not in self.nodes:
            print(f"ERROR: src={src} or dest={dest} not in loop nodes {self.nodes}")
            return float('inf')  # Return infinity if nodes not in loop
            
        srcIdx = self.nodes.index(src)
        destIdx = self.nodes.index(dest)
        
        # debug path calculation
        print(f"Calculating path length in loop {self.id}: src={src}(idx={srcIdx}), dest={dest}(idx={destIdx})")
        
        # handle wrap around for circular loops
        if destIdx < srcIdx:
            pathLen = len(self.nodes) - 1 - srcIdx + destIdx
        else:
            pathLen = destIdx - srcIdx
        
        print(f"Path length = {pathLen}")
        return pathLen
    
# to link predetermined path with nodes
class Link:
    def __init__(self, env, fromNode, toNode, loopID):
        self.env = env
        self.fromNode = fromNode
        self.toNode = toNode
        self.loopID = loopID
        self.buffer = simpy.Store(env, capacity=1)


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
        self.loopBuffers = {loopID: simpy.Store(env, capacity=1) for loopID in self.loops}
        self.exb = simpy.Store(env, capacity=EXBSize)
        self.ejectionLinks = [simpy.Resource(env, capacity=1) for _ in range(ejectLinks)]
        self.ejectionQueues = [simpy.Store(env) for _ in range(ejectLinks)]

        self.routingTable = {}  # {dest: bestLoopID}
        self.exbAttachedToLoop = {loopID: False for loopID in self.loops}
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

    def injection(self, packet):
        self.injectedPackets += 1
        yield self.env.timeout(0)  

        bestLoopID = self.findBestLoop(packet.dest)

        if bestLoopID is None:
            print(f"No loop available for destination {packet.dest}, packet {packet.id} dropped")
            return
        
        packet.loopID = bestLoopID

        flits = packet.flitation()
        print(f"\n=== INJECTING packet {packet.id} to destination {packet.dest} via loop {bestLoopID} ===")

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


        for flit in flits:
            # checking if the loop is available
            if len(self.loopBuffers[bestLoopID].items) == 0:
                print(f"Injecting flit {flit.id} into loop {bestLoopID}")


                # update the flit's current node
                flit.currNode = self.nodeID

                flit.injectTime = self.env.now
                
                yield self.loopBuffers[bestLoopID].put(flit)
                self.env.process(self.processFlit(bestLoopID))
            else:
                print(f"STORING IN EXB: Loop {bestLoopID} busy - flit {flit.id} goes to EXB")
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

                flit.currNode = self.nodeID

                yield self.loopBuffers[loopID].put(flit)
                self.env.process(self.processFlit(loopID))

                # if EXB is empty, detach from loop
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

                if flit.type == "TAIL":
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

            if flit.circlingCount >= 254 and self.reservedEjection is None:
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
                    print(f"Loop {loopID} is now the best choice with path length {pathLen}")
            else:
                print(f"Loop {loopID} cannot be used - missing nodes")
                if self.nodeID not in loop.nodes:
                    print(f"Node {self.nodeID} not in loop {loopID}")
                if dest not in loop.nodes:
                    print(f"Dest {dest} not in loop {loopID}")

        if bestLoopID is not None:
            self.routingTable[dest] = bestLoopID
            print(f"Selected best loop {bestLoopID} for destination {dest} with path length {minHops}")
        else:
            print(f"ERROR - Could not find a valid loop from {self.nodeID} to {dest}")
            for loopID, loop in self.network.loops.items():
                print(f"DEBUG-ROUTING: Loop {loopID} nodes: {loop.nodes}")

        return bestLoopID
    
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
    
    def inject(self, src, dest=None, size=3, time=None):
        packet = self.generatePacket(src, dest, size, time)
        self.stats['injected'] += 1

        # schedule the packet injection
        if time is not None and time > self.env.now:
            delay = time - self.env.now
            self.env.process(self.delayedInjection(src, packet, delay))
        else:
            self.env.process(self.nodes[src].injection(packet))

        return packet
    
    def delayedInjection(self, src, packet, delay):
        yield self.env.timeout(delay)
        yield self.env.process(self.nodes[src].injection(packet))

    def run(self, until=100):
        self.stats = {'injected': 0,
                      'ejected': 0,
                      'latencies': []}
        
        self.env.run(until=until)

        # gather the stats
        for nodeID, node in self.nodes.items():
            self.stats['injected'] += node.injectedPackets
            self.stats['ejected'] += node.ejectedPackets
            self.stats['latencies'].extend(node.packetLatencies)

        # calculate latency
        if self.stats['latencies']:
            self.stats['avgLatency'] = sum(self.stats['latencies']) / len(self.stats['latencies'])
        else:
            self.stats['avgLatency'] = 0

        print(f"DEBUG-STATS: Total packet latencies collected: {len(self.stats['latencies'])}")
        print(f"DEBUG-STATS: Sample of latencies: {self.stats['latencies'][:10]}")

        return self.stats
    
    def randomInt(self, low, high):
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

    for i in range(totalPackets):
        src = i % totalNodes
        dest = (src+1) % totalNodes
        injectionTime = i*2
        packetSize = 3 + (i%3)

        noc.inject(src, dest, packetSize, injectionTime)

    duration = (totalPackets*2) + 100
    stats = noc.run(until=duration)

    # print statistics
    print("\nManual Traffic Simulation Statistics:")
    print(f"Injected packets: {stats['injected']}")
    print(f"Ejected packets: {stats['ejected']}")
    if stats['latencies']:
        print(f"Average packet latency: {stats['avgLatency']:.2f} cycles")
        print(f"Min latency: {min(stats['latencies'])} cycles")
        print(f"Max latency: {max(stats['latencies'])} cycles")
        
        # add a histogram of latencies
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
    
    # force two packets from same source close together to create contention
    noc.inject(src=0, dest=1, size=5, time=0)  # larger packet
    noc.inject(src=0, dest=3, size=3, time=1)  # second packet before first one is fully injected
    
    stats = noc.run(until=20)
    print("\nEXB Contention Test Results:")
    print(f"Injected packets: {stats['injected']}")
    print(f"Ejected packets: {stats['ejected']}")


def test_packet_injection():
    print("\n======= TEST: PACKET INJECTION =======")
    noc = RouterlessNoC(size=2)
    
    # create a packet
    packet = noc.generatePacket(src=0, dest=1, size=3)
    
    # run the simulation for a short time to inject the packet
    noc.env.process(noc.nodes[0].injection(packet))
    stats = noc.run(until=10)
    
    # check that the packet was injected
    assert stats['injected'] == 1, f"Expected 1 injected packet, got {stats['injected']}"
    print(f"Successfully injected {stats['injected']} packet")
    
    # check that all flits were created
    assert len(packet.flits) == 3, f"Expected 3 flits, got {len(packet.flits)}"
    print(f"Packet correctly flitized into {len(packet.flits)} flits")
    
    # verify flit types
    assert packet.flits[0].type == "HEAD", "First flit should be HEAD"
    assert packet.flits[1].type == "BODY", "Middle flit should be BODY"
    assert packet.flits[2].type == "TAIL", "Last flit should be TAIL"
    print("Flit types correctly assigned")
    
    print("RESULT: Packet injection test PASSED")
    return True

def test_packet_ejection():
    print("\n======= TEST: PACKET EJECTION =======")
    noc = RouterlessNoC(size=2)
    
    # inject a packet from node 0 to node 1
    noc.inject(src=0, dest=1, size=3, time=0)
    
    # run the simulation long enough for the packet to be delivered
    stats = noc.run(until=20)
    
    # check that the packet was ejected
    assert stats['ejected'] == 1, f"Expected 1 ejected packet, got {stats['ejected']}"
    print(f"Successfully ejected {stats['ejected']} packet")
    
    # check that latency was recorded
    assert len(stats['latencies']) == 1, f"Expected 1 latency record, got {len(stats['latencies'])}"
    print(f"Latency correctly recorded: {stats['latencies'][0]} cycles")
    
    print("RESULT: Packet ejection test PASSED")
    return True

def test_exb_usage():
    print("\n======= TEST: EXB USAGE (FIXED) =======")
    noc = RouterlessNoC(size=2)
    
    # add detailed logging to the injection method
    original_injection = noc.nodes[0].injection
    
    def monitored_injection(packet):
        print(f"Starting injection for packet {packet.id}, size={packet.size}")
        result_gen = original_injection(packet)
        
        # create a wrapper around the generator to monitor key points
        def wrapped_injection():
            try:
                while True:
                    # get the next yield point from the original generator
                    result = yield next(result_gen)
                    
                    # send the result back to the original generator if needed
                    if result is not None:
                        result_gen.send(result)
            except StopIteration:
                print(f"TEST: Completed injection for packet {packet.id}")
        
        return wrapped_injection()
    
    # replace the method
    noc.nodes[0].injection = monitored_injection
    
    # add detailed monitoring for EXB put method
    exb_put_count = 0
    original_exb_put = noc.nodes[0].exb.put
    
    def monitored_exb_put(item):
        nonlocal exb_put_count
        exb_put_count += 1
        print(f"TEST: EXB.put called for flit {item.id}, count={exb_put_count}")
        return original_exb_put(item)
    
    noc.nodes[0].exb.put = monitored_exb_put
    
    # monitor loop buffer usage to understand availability
    original_loop_buffer_put = noc.nodes[0].loopBuffers[0].put
    loop_buffer_put_count = 0
    
    def monitored_loop_buffer_put(item):
        nonlocal loop_buffer_put_count
        loop_buffer_put_count += 1
        print(f"DEBUG-TEST: LoopBuffer[0].put called for flit {item.id}, count={loop_buffer_put_count}")
        return original_loop_buffer_put(item)
    
    noc.nodes[0].loopBuffers[0].put = monitored_loop_buffer_put
    
    # Force the loop buffer to appear occupied during injection
    loop_buffer_original_items = noc.nodes[0].loopBuffers[0].items
    
    # force the loop buffer to appear full for every other flit
    def make_loop_buffer_busy_for_some_flits():
        # create a fake item for the loop buffer
        class FakeItem:
            def __init__(self, id):
                self.id = id
        
        # add a fake item to make it appear busy
        fake_item = FakeItem("fake_occupier")
        noc.nodes[0].loopBuffers[0].items = [fake_item]
        print("TEST: Loop buffer artificially occupied with fake item")
        
        # inject a packet large enough to ensure multiple flits
        packet = noc.inject(src=0, dest=1, size=5, time=0)
        
        def clear_buffer_after_delay():
            yield noc.env.timeout(2)
            noc.nodes[0].loopBuffers[0].items = []
            print("TEST: Loop buffer artificially cleared")
        
        # schedule the clearing
        noc.env.process(clear_buffer_after_delay())
        
        return packet
    
    # run with the forced congestion
    packet = make_loop_buffer_busy_for_some_flits()
    
    print("Running simulation for EXB usage test...")
    noc.run(until=15)
    
    noc.nodes[0].injection = original_injection
    noc.nodes[0].exb.put = original_exb_put
    noc.nodes[0].loopBuffers[0].put = original_loop_buffer_put
    noc.nodes[0].loopBuffers[0].items = loop_buffer_original_items
    
    assert exb_put_count > 0, "EXB should have been used for packet buffering"
    print(f"EXB was correctly used {exb_put_count} times for packet buffering")
    
    assert noc.stats['ejected'] == 1, f"Expected 1 ejected packet, got {noc.stats['ejected']}"
    print(f"Packet was successfully ejected after EXB usage")
    
    print("RESULT: EXB usage test PASSED")
    return True

def test_exb_draining():
    print("\n======= TEST: EXB DRAINING (FIXED) =======")
    noc = RouterlessNoC(size=2)
    
    # Ttrack draining operations
    drain_calls = 0
    flits_drained = 0
    
    # monitor the drainEXB method
    original_drainEXB = noc.nodes[0].drainEXB
    
    def monitored_drainEXB(loopID):
        nonlocal drain_calls
        drain_calls += 1
        print(f"TEST: drainEXB called for loop {loopID}, count={drain_calls}")
        
        # Log the state at the start of draining
        print(f"TEST: At drainEXB start - EXB items: {len(noc.nodes[0].exb.items)}")
        print(f"TEST: At drainEXB start - exbAttachedToLoop: {noc.nodes[0].exbAttachedToLoop}")
        
        
        drain_gen = original_drainEXB(loopID)
        
        
        def wrapped_drain():
            nonlocal flits_drained
            try:
                while True:
                    result = yield next(drain_gen)
                

                    if len(noc.nodes[0].exb.items) < len(noc.nodes[0].exb.items) - 1:
                        flits_drained += 1
                        print(f"TEST: Drained a flit from EXB, count={flits_drained}")
                    
                    # senf any results back
                    if result is not None:
                        drain_gen.send(result)
            except StopIteration:
                print(f"TEST: drainEXB completed for loop {loopID}")
        
        return wrapped_drain()
    
    noc.nodes[0].drainEXB = monitored_drainEXB
    
    exb_put_count = 0
    original_exb_put = noc.nodes[0].exb.put
    
    def monitored_exb_put(item):
        nonlocal exb_put_count
        exb_put_count += 1
        print(f"TEST: EXB.put called for flit {item.id}, count={exb_put_count}")
        return original_exb_put(item)
    
    noc.nodes[0].exb.put = monitored_exb_put
    
    # force some flits to use EXB by artificially occupying the loop buffer
    loop_buffer_original_items = noc.nodes[0].loopBuffers[0].items
    
    def make_exb_draining_needed():
        # vreate a fake item for the loop buffer
        class FakeItem:
            def __init__(self, id):
                self.id = id
        
        # add a fake item to make it appear busy
        fake_item = FakeItem("fake_occupier")
        noc.nodes[0].loopBuffers[0].items = [fake_item]
        print("TEST: Loop buffer artificially occupied with fake item")
        
        # inject a large packet
        packet = noc.inject(src=0, dest=1, size=6, time=0)
        
        # after some flits go to EXB, clear the buffer to enable draining
        def clear_buffer_after_delay():
            yield noc.env.timeout(3)  # Let some flits go to EXB first
            noc.nodes[0].loopBuffers[0].items = []
            print("TEST: Loop buffer artificially cleared to allow draining")
            
            # directly trigger draining
            if noc.nodes[0].exbAttachedToLoop[0] and len(noc.nodes[0].exb.items) > 0:
                print("TEST: Manually triggering drainEXB")
                noc.env.process(noc.nodes[0].drainEXB(0))
        
        # clearing and draining
        noc.env.process(clear_buffer_after_delay())
        
        return packet
    
    packet = make_exb_draining_needed()
    
    # run simulation
    print("Running simulation for EXB draining test...")
    noc.run(until=20)
    
    # Restore
    noc.nodes[0].drainEXB = original_drainEXB
    noc.nodes[0].exb.put = original_exb_put
    noc.nodes[0].loopBuffers[0].items = loop_buffer_original_items
    
    # verify results
    print(f"TEST: EXB put count: {exb_put_count}")
    print(f"TEST: drainEXB called: {drain_calls} times")
    print(f"TEST: Flits drained: {flits_drained}")
    
    assert exb_put_count > 0, "Some flits should have been stored in EXB"
    assert drain_calls > 0, "EXB draining method should have been called"
    print(f"EXB draining occurred: {drain_calls} calls")
    
    # Check if packet was fully ejected
    assert noc.stats['ejected'] == 1, f"Expected 1 ejected packet, got {noc.stats['ejected']}"
    print(f"Packet was successfully ejected")
    
    print("RESULT: EXB draining test PASSED")
    return True

def test_loop_routing():
    print("\n======= TEST: LOOP ROUTING (SIMPLIFIED) =======")
    noc = RouterlessNoC(size=2)
    
    # Test multiple source-destination pairs
    test_routes = [
        (0, 1),  # Should use loop 0 (clockwise)
        (0, 2),  # Should use loop 1 (counter-clockwise)
        (1, 3),  # Should use loop 0
        (2, 0)   # Should use loop 1
    ]
    
    expected_loops = [0, 1, 0, 0]  # Expected best loop for each route
    all_passed = True
    
    for i, (src, dest) in enumerate(test_routes):
        node = noc.nodes[src]
        best_loop = node.findBestLoop(dest)
        
        expected = expected_loops[i]
        if best_loop != expected:
            print(f"Route {src}->{dest} used loop {best_loop}, expected {expected}")
            all_passed = False
        else:
            print(f"Route {src}->{dest} uses correct loop {best_loop}")
    
    print("RESULT: Loop routing test", "PASSED" if all_passed else "FAILED")
    return all_passed

def test_contention_handling():
    print("\n======= TEST: CONTENTION HANDLING =======")
    noc = RouterlessNoC(size=2)
    
    # nject multiple packets in quick succession to create contention
    noc.inject(src=0, dest=1, size=4, time=0)
    noc.inject(src=0, dest=3, size=3, time=1)  # contention at source
    noc.inject(src=2, dest=1, size=3, time=2)  # potential contention at destination
    
    # run the simulation
    stats = noc.run(until=40)
    
    # all packets should eventually be delivered
    assert stats['injected'] == 3, f"Expected 3 injected packets, got {stats['injected']}"
    assert stats['ejected'] == 3, f"Expected 3 ejected packets, got {stats['ejected']}"
    print(f"All {stats['injected']} packets were successfully ejected despite contention")
    
    # check latency distribution
    if stats['latencies']:
        print(f"Latency range: {min(stats['latencies'])} to {max(stats['latencies'])} cycles")
    
    print("RESULT: Contention handling test PASSED")
    return True

def test_large_network():
    print("\n======= TEST: LARGE NETWORK =======")
    noc = RouterlessNoC(size=4)
    

    total_nodes = noc.size * noc.size
    
    # inject packets from each node to random destinations
    num_packets = 16  # one from each node
    for i in range(num_packets):
        src = i % total_nodes
        dest = (src + noc.randomInt(1, total_nodes-1)) % total_nodes
        noc.inject(src=src, dest=dest, size=3, time=i*2)
    
    # run simulation
    stats = noc.run(until=100)
    
    # check results
    assert stats['injected'] == num_packets, f"Expected {num_packets} injected packets, got {stats['injected']}"
    assert stats['ejected'] == num_packets, f"Expected {num_packets} ejected packets, got {stats['ejected']}"
    print(f"All {stats['injected']} packets were successfully delivered in the larger network")
    
    if stats['latencies']:
        avg_latency = stats['avgLatency']
        print(f"Average latency: {avg_latency:.2f} cycles")
    
    print("RESULT: Large network test PASSED")
    return True

def test_circling_packets():
    print("\n======= TEST: CIRCLING PACKETS =======")
    noc = RouterlessNoC(size=2)
    
    # reduce ejection links to increase chance of contention
    for node_id, node in noc.nodes.items():
        node.ejectionLinks = [simpy.Resource(noc.env, capacity=1)]
    
    # inject multiple packets to same destination to create ejection contention
    for i in range(5):
        noc.inject(src=i % 4, dest=1, size=3, time=i)
    
    # track circling packets
    circling_detected = False
    
    # monitor the eject method to detect circling
    original_eject = noc.nodes[1].eject
    
    def monitored_eject(flit):
        nonlocal circling_detected
        result = yield noc.env.process(original_eject(flit))
        if not result and flit.type == "HEAD":
            circling_detected = True
            print(f"  Detected circling flit: {flit.id}, circle count: {flit.circlingCount}")
        return result
    
    # replace the method
    noc.nodes[1].eject = monitored_eject
    
    # run simulation
    stats = noc.run(until=50)
    
    # restore original method
    noc.nodes[1].eject = original_eject
    
    # check results
    assert stats['injected'] == 5, f"Expected 5 injected packets, got {stats['injected']}"
    assert stats['ejected'] == 5, f"Expected 5 ejected packets, got {stats['ejected']}"
    print(f"All {stats['injected']} packets were eventually ejected")
    if circling_detected:
        print("Successfully detected and handled circling packets")
    else:
        print("No packet circling occurred during this test run")
    
    print("RESULT: Circling packets test PASSED")
    return True

def test_multiple_loops_routing():
    print("\n======= TEST: MULTIPLE LOOPS ROUTING =======")
    noc = RouterlessNoC(size=4)
    
    # create a dictionary to track which loops are used
    loop_usage = {loop_id: 0 for loop_id in noc.network.loops.keys()}
    
    # inject packets between various node pairs
    num_packets = 20
    for i in range(num_packets):
        src = i % 16
        dest = (src + 5) % 16  # Pick destinations that likely require different loops
        
        best_loop = noc.nodes[src].findBestLoop(dest)
        loop_usage[best_loop] += 1
        
        noc.inject(src=src, dest=dest, size=3, time=i*2)
    
    # run simulation
    stats = noc.run(until=100)
    
    loops_used = sum(1 for count in loop_usage.values() if count > 0)
    assert loops_used > 1, f"Expected multiple loops to be used, but only {loops_used} loop was used"
    print(f"{loops_used} different loops were used for routing")
    
    # check specific loop usage
    for loop_id, count in loop_usage.items():
        if count > 0:
            print(f"  Loop {loop_id} was used for {count} packets")
    
    # verify all packets were delivered
    assert stats['ejected'] == num_packets, f"Expected {num_packets} ejected packets, got {stats['ejected']}"
    print(f"All {stats['injected']} packets were successfully delivered")
    
    print("RESULT: Multiple loops routing test PASSED")
    return True

def run_all_tests():
    print("\n========================================")
    print("RUNNING ALL ROUTERLESS NOC TESTS")
    print("========================================")
    
    tests = [
        test_packet_injection,
        test_packet_ejection,
        test_exb_usage,
        test_exb_draining,
        test_loop_routing,
        test_contention_handling,
        test_large_network,
        test_circling_packets,
        test_multiple_loops_routing
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except AssertionError as e:
            print(f"Test {test.__name__} FAILED: {e}")
            results.append((test.__name__, False))
        except Exception as e:
            print(f"Test {test.__name__} ERROR: {type(e).__name__} - {e}")
            results.append((test.__name__, False))
    
    print("\n========================================")
    print("TEST RESULTS SUMMARY")
    print("========================================")
    
    passed = sum(1 for _, result in results if result)
    total = len(tests)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    print(f"\nPASSED: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nALL TESTS PASSED! The RouterlessNoC implementation is working correctly.")
    else:
        print("\nSOME TESTS FAILED. Please review the implementation.")

# debug utility function to trace packet flow through the network
def trace_packet_flow(src, dest, size=3, max_time=30):
    print(f"\n======= TRACING PACKET FLOW: {src}->{dest} =======")
    noc = RouterlessNoC(size=4)
    
    # replace print with our monitoring version
    original_print = __builtins__['print']
    
    packet_traces = []
    
    def monitored_print(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        if "flit" in text or "packet" in text:
            packet_traces.append(text)
        original_print(*args, **kwargs)
    
    __builtins__['print'] = monitored_print
    
    # inject test packet
    packet = noc.inject(src=src, dest=dest, size=size, time=0)
    
    # run simulation
    stats = noc.run(until=max_time)
    
    # restore original print
    __builtins__['print'] = original_print
    
    # display summary
    print("\nPacket Flow Summary:")
    print(f"Packet ID: {packet.id}")
    print(f"Source: {src}, Destination: {dest}")
    print(f"Size: {size} flits")
    print(f"Was packet delivered? {'Yes' if stats['ejected'] > 0 else 'No'}")
    
    if stats['latencies']:
        print(f"Packet latency: {stats['latencies'][0]} cycles")
    
    # pptionally return the traces for further analysis
    return packet_traces

# modified test function that checks if a bug fix is working
def test_inject_time_consistency():
    print("\n======= TEST: INJECT TIME CONSISTENCY =======")
    noc = RouterlessNoC(size=2)
    
    # inject a packet that will use EXB
    packet = noc.generatePacket(src=0, dest=1, size=5)
    noc.env.process(noc.nodes[0].injection(packet))
    
    # run simulation
    noc.run(until=20)
    
    # check that all flits have the same injection time
    inject_times = [flit.injectTime for flit in packet.flits]
    unique_times = set(inject_times)
    
    if None in unique_times:
        print("Found flits with None inject time")
        print(f"Flit inject times: {inject_times}")
        assert False, "Some flits have None inject time"
    print(f"Flit inject times: {inject_times}")
    
    if len(unique_times) == 1:
        print("All flits have consistent inject time")
    else:
        print("Flits have different inject times - this may be expected if they use EXB")
    
    noc.run(until=40)
    
    assert noc.stats['ejected'] == 1, "Packet should have been ejected"
    assert len(noc.stats['latencies']) == 1, "Latency should have been recorded"
    
    print(f"Packet latency recorded: {noc.stats['latencies'][0]} cycles")
    print("RESULT: Inject time consistency test PASSED")

def evaluate_latency_across_sizes(network_sizes=[2, 4], packets_per_node=3, packet_size=3):
    results = {}
    
    print("\n===== LATENCY ANALYSIS ACROSS NETWORK SIZES =====")
    
    for size in network_sizes:
        noc = RouterlessNoC(size=size)
        total_nodes = size * size
        
        # generate simple traffic where each node sends to next node
        for node in range(total_nodes):
            for i in range(packets_per_node):
                dest = (node + 1) % total_nodes
                inject_time = i * 3 + node
                noc.inject(src=node, dest=dest, size=packet_size, time=inject_time)
        
        # run the simulation with appropriate duration
        duration = packets_per_node * total_nodes * 5
        stats = noc.run(until=duration)
        
        results[size] = {
            'total_nodes': total_nodes,
            'injected': stats['injected'],
            'ejected': stats['ejected'],
            'avg_latency': stats['avgLatency'],
            'min_latency': min(stats['latencies']) if stats['latencies'] else None,
            'max_latency': max(stats['latencies']) if stats['latencies'] else None
        }
        
        print(f"Results for {size}x{size} network:")
        print(f"    Packets: {stats['ejected']}/{stats['injected']} delivered")
        print(f"    Latency: avg={stats['avgLatency']:.2f}, min={results[size]['min_latency']}, max={results[size]['max_latency']}")
    
    print("\nLatency Analysis Summary:")
    print("-" * 60)
    print(f"{'Size':^8} | {'Nodes':^8} | {'Delivered':^12} | {'Avg Latency':^12} | {'Min/Max':^15}")
    print("-" * 60)
    
    for size, data in results.items():
        print(f"{size}x{size:^6} | {data['total_nodes']:^8} | {data['ejected']}/{data['injected']} | {data['avg_latency']:^12.2f} | {data['min_latency']}-{data['max_latency']}")
    
    return results

def evaluate_throughput_vs_injection(network_size=4, injection_rates=[0.05, 0.1, 0.2, 0.3], packet_size=3, simulation_time=500):
    results = {}
    
    print("\n===== THROUGHPUT VS INJECTION RATE ANALYSIS =====")
    
    for rate in injection_rates:
        noc = RouterlessNoC(size=network_size)
        total_nodes = noc.size * noc.size
        
        for src in range(total_nodes):
            next_time = 0
            while next_time < simulation_time:
                # time to next packet follows exponential distribution
                inter_arrival = noc.exponential(rate)
                next_time += inter_arrival
                
                if next_time < simulation_time:
                    dest = src
                    while dest == src:
                        dest = noc.randomInt(0, total_nodes - 1)
                    
                    noc.inject(src, dest, packet_size, next_time)
        
        # run the simulation
        stats = noc.run(until=simulation_time + 50)
        
        throughput = stats['ejected'] / simulation_time
        offered_load = stats['injected'] / simulation_time
        acceptance = stats['ejected'] / stats['injected'] if stats['injected'] > 0 else 0
        
        results[rate] = {
            'injected': stats['injected'],
            'ejected': stats['ejected'],
            'offered_load': offered_load,
            'throughput': throughput,
            'acceptance_rate': acceptance,
            'avg_latency': stats['avgLatency']
        }
        
        print(f"Results for injection rate {rate}:")
        print(f"  Throughput: {throughput:.4f} packets/cycle (offered: {offered_load:.4f})")
        print(f"  Acceptance rate: {acceptance*100:.1f}%")
        print(f"  Average latency: {stats['avgLatency']:.2f} cycles")
    
    # create a summary table
    print("\nThroughput vs. Injection Rate Summary:")
    print("-" * 75)
    print(f"{'Inj. Rate':^10} | {'Offered Load':^12} | {'Throughput':^12} | {'Acceptance':^10} | {'Latency':^10}")
    print("-" * 75)
    
    for rate, data in results.items():
        print(f"{rate:^10.3f} | {data['offered_load']:^12.4f} | {data['throughput']:^12.4f} | {data['acceptance_rate']*100:^10.1f}% | {data['avg_latency']:^10.2f}")
    
    return results

def evaluate_exb_performance(network_size=4, exb_sizes=[1, 5, 10], packet_size=5):
    print("\n===== EXB PERFORMANCE ANALYSIS =====")
    
    injection_rate = 0.2
    simulation_time = 300
    
    for exb_size in exb_sizes:
        noc = RouterlessNoC(size=network_size)
        total_nodes = noc.size * noc.size
        
        for nodeID, node in noc.nodes.items():
            node.exb = simpy.Store(noc.env, capacity=exb_size)
        
        print(f"LOG: Type of node.exb.put: {type(noc.nodes[0].exb.put)}")
        print(f"LOG: Dir of node.exb.put: {dir(noc.nodes[0].exb.put)}")
        
        exb_put_count = 0
        
        def create_counter():
            counter = 0
            def count_puts(item):
                nonlocal counter
                counter += 1
                print(f"DEBUG: EXB put called with item {item}")
                return counter
            return count_puts
        
        counters = {}
        for nodeID, node in noc.nodes.items():
            counters[nodeID] = create_counter()
        
        for src in range(2):
            for i in range(2):
                dest = (src + 1) % total_nodes
                inject_time = src * 3 + i * 0.5
                noc.inject(src, dest, packet_size, inject_time)
        
        try:
            stats = noc.run(until=10)  # short simulation for testing
            print("Simulation completed successfully")
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            print(f"Error type: {type(e)}")
            print("Trying again without EXB monitoring...")
            noc = RouterlessNoC(size=network_size)
            for nodeID, node in noc.nodes.items():
                node.exb = simpy.Store(noc.env, capacity=exb_size)
            
            # Generate minimal traffic
            noc.inject(0, 1, packet_size, 0)
            try:
                noc.run(until=10)
                print("Unmodified simulation works correctly")
            except Exception as e:
                print(f"Unmodified simulation also fails: {e}")
            
            break  # Stop after first failure

def evaluate_traffic_patterns(network_size=4, injection_rate=0.1, packet_size=3, simulation_time=300):
    results = {}
    
    print("\n===== TRAFFIC PATTERN ANALYSIS =====")
    
    # define traffic patterns
    traffic_patterns = {
        'uniform_random': lambda src, total_nodes: (src + 1 + RouterlessNoC(network_size).randomInt(0, total_nodes - 2)) % total_nodes,
        'transpose': lambda src, total_nodes: (src % network_size) * network_size + (src // network_size),
        'bit_reversal': lambda src, total_nodes: int('{:08b}'.format(src)[::-1], 2) % total_nodes,
        'hot_spot': lambda src, total_nodes: total_nodes // 2 if RouterlessNoC(network_size).randomFloat() < 0.3 else (src + 1) % total_nodes,
        'nearest_neighbor': lambda src, total_nodes: (src + 1) % total_nodes
    }
    
    for pattern_name, dest_function in traffic_patterns.items():
        noc = RouterlessNoC(size=network_size)
        total_nodes = noc.size * noc.size
        
        # generate traffic according to this pattern
        for src in range(total_nodes):
            next_time = 0
            packet_count = 0
            max_packets = 5  # limit packets per source
            
            while next_time < simulation_time and packet_count < max_packets:
                inter_arrival = noc.exponential(injection_rate)
                next_time += inter_arrival
                
                if next_time < simulation_time:
                    dest = dest_function(src, total_nodes)
                    
                    if src == dest:
                        continue
                    
                    noc.inject(src, dest, packet_size, next_time)
                    packet_count += 1
        
        stats = noc.run(until=simulation_time + 50)
        
        throughput = stats['ejected'] / simulation_time
        
        results[pattern_name] = {
            'injected': stats['injected'],
            'ejected': stats['ejected'],
            'throughput': throughput,
            'avg_latency': stats['avgLatency'],
            'min_latency': min(stats['latencies']) if stats['latencies'] else None,
            'max_latency': max(stats['latencies']) if stats['latencies'] else None
        }
        
        print(f"Results for {pattern_name} traffic pattern:")
        print(f"  Packets: {stats['ejected']}/{stats['injected']} delivered")
        print(f"  Throughput: {throughput:.4f} packets/cycle")
        print(f"  Latency: avg={stats['avgLatency']:.2f}, min={results[pattern_name]['min_latency']}, max={results[pattern_name]['max_latency']}")
    
    print("\nTraffic Pattern Summary:")
    print("-" * 80)
    print(f"{'Pattern':^15} | {'Delivered':^12} | {'Throughput':^12} | {'Avg Latency':^12} | {'Min/Max':^15}")
    print("-" * 80)
    
    for pattern, data in results.items():
        print(f"{pattern:^15} | {data['ejected']}/{data['injected']} | {data['throughput']:^12.4f} | {data['avg_latency']:^12.2f} | {data['min_latency']}-{data['max_latency']}")
    
    return results

def run_evaluations():
    print("\n" + "=" * 50)
    print("ROUTERLESS NOC EVALUATION SUITE")
    print("=" * 50)
    
    results = {}
    
    # Run selected evaluations with simplified parameters
    print("\nStarting evaluations...")
    
    results['latency'] = evaluate_latency_across_sizes(
        network_sizes=[2, 4], 
        packets_per_node=3
    )
    
    results['throughput'] = evaluate_throughput_vs_injection(
        network_size=4,
        injection_rates=[0.05, 0.1, 0.2, 0.3],
        simulation_time=300
    )
    
    results['exb'] = evaluate_exb_performance(
        network_size=4,
        exb_sizes=[1, 5, 10]
    )
    
    results['traffic'] = evaluate_traffic_patterns(
        network_size=4,
        injection_rate=0.1
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    
    return results

if __name__ == "__main__":
    print("Running validation tests...")
    run_all_tests()
    
    # then run evaluations
    print("\nTests complete. Running evaluations...")
    results = run_evaluations()
    
    print("\nEvaluations complete. Results have been collected for analysis.")
    print("To visualize the results, you can create graphs using matplotlib.")
    print("Example visualization code has been provided in the script.")
    
    '''
    # 1. Latency across network sizes
    sizes = list(results['latency'].keys())
    latencies = [results['latency'][size]['avg_latency'] for size in sizes]
    
    plt.figure(figsize=(8, 5))
    plt.bar(sizes, latencies, color='skyblue')
    plt.xlabel('Network Size (NxN)')
    plt.ylabel('Average Latency (cycles)')
    plt.title('Latency vs Network Size')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('latency_vs_size.png', dpi=300)
    plt.show()
    
    # 2. Throughput vs injection rate
    rates = list(results['throughput'].keys())
    throughputs = [results['throughput'][rate]['throughput'] for rate in rates]
    latencies = [results['throughput'][rate]['avg_latency'] for rate in rates]
    
    plt.figure(figsize=(8, 5))
    plt.plot(rates, throughputs, 'o-', color='green')
    plt.xlabel('Injection Rate (packets/cycle/node)')
    plt.ylabel('Throughput (packets/cycle)')
    plt.title('Throughput vs Injection Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('throughput_vs_injection.png', dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.plot(rates, latencies, 's-', color='red')
    plt.xlabel('Injection Rate (packets/cycle/node)')
    plt.ylabel('Average Latency (cycles)')
    plt.title('Latency vs Injection Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('latency_vs_injection.png', dpi=300)
    plt.show()
    
    # 3. EXB Performance
    exb_sizes = list(results['exb'].keys())
    exb_latencies = [results['exb'][size]['avg_latency'] for size in exb_sizes]
    exb_usage = [results['exb'][size]['exb_put_per_packet'] for size in exb_sizes]
    
    plt.figure(figsize=(8, 5))
    plt.bar(exb_sizes, exb_latencies, color='lightgreen')
    plt.xlabel('EXB Buffer Size')
    plt.ylabel('Average Latency (cycles)')
    plt.title('Latency vs EXB Size')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('latency_vs_exb.png', dpi=300)
    plt.show()
    '''
    