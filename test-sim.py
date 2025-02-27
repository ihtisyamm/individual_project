import simpy
import math
import time as time_module  # Use standard library time module

class Flit:
    """A single flow control unit (flit) that makes up a packet"""
    def __init__(self, id, pid, type, src, dest, time, loop_id=None):
        self.id = id
        self.pid = pid      # packet id
        self.type = type    # HEAD | BODY | TAIL
        self.src = src
        self.dest = dest
        self.timestamp = time
        self.loop_id = loop_id  # The loop this flit is assigned to
        self.circling_count = 0  # For livelock avoidance
        self.current_node = src  # Track the current node the flit is at
        self.injection_time = None  # When the packet was injected
        self.ejection_time = None   # When the packet was ejected (to track latency)

class Packet:
    """A packet to be sent in the network"""
    def __init__(self, id, src, dest, size, time):
        self.id = id
        self.src = src
        self.dest = dest
        self.size = size
        self.timestamp = time
        self.flits = []
        self.loop_id = None  # The loop this packet will be assigned to

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
                time=self.timestamp,
                loop_id=self.loop_id
            )
            self.flits.append(flit)
        return self.flits

class Loop:
    """Represents a loop in the network"""
    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes  # List of nodes in this loop, in order

    def __repr__(self):
        return f"Loop {self.id}: {' -> '.join([str(n) for n in self.nodes])}"
    
    def get_next_node(self, current_node):
        """Get the next node in the loop after the current node"""
        if current_node not in self.nodes:
            return None
            
        idx = self.nodes.index(current_node)
        next_idx = (idx + 1) % (len(self.nodes) - 1)  # Last node is same as first
        return self.nodes[next_idx]
    
    def get_path_length(self, src, dest):
        """Get the number of hops from src to dest along this loop"""
        if src not in self.nodes or dest not in self.nodes:
            return float('inf')
            
        src_idx = self.nodes.index(src)
        dest_idx = self.nodes.index(dest)
        
        # If dest is behind src in the loop, we need to wrap around
        if dest_idx < src_idx:
            return len(self.nodes) - 1 - src_idx + dest_idx
        
        return dest_idx - src_idx

class Link:
    """Represents a physical link between two nodes for a specific loop"""
    def __init__(self, env, from_node, to_node, loop_id):
        self.env = env
        self.from_node = from_node
        self.to_node = to_node
        self.loop_id = loop_id
        self.buffer = simpy.Store(env, capacity=1)  # Links have 1-flit capacity
    
    def __repr__(self):
        return f"Link({self.from_node}->{self.to_node}, loop={self.loop_id})"

class RoutelessNetwork:
    """Models the entire network with physical links between nodes"""
    def __init__(self, env, loops):
        self.env = env
        self.loops = {loop.id: loop for loop in loops}
        self.links = {}  # Dictionary of links (from_node, to_node, loop_id) -> Link
        
        # Create all the physical links based on the loops
        self.create_links()
    
    def create_links(self):
        """Create physical links between nodes based on the loops"""
        for loop_id, loop in self.loops.items():
            # For each consecutive pair of nodes in the loop
            for i in range(len(loop.nodes) - 1):
                from_node = loop.nodes[i]
                to_node = loop.nodes[i + 1]
                link = Link(self.env, from_node, to_node, loop_id)
                self.links[(from_node, to_node, loop_id)] = link
    
    def transmit(self, from_node, to_node, loop_id, flit):
        """Transmit a flit from one node to another on a specific loop"""
        link = self.links.get((from_node, to_node, loop_id))
        if link:
            return link.buffer.put(flit)
        else:
            raise ValueError(f"No link from {from_node} to {to_node} on loop {loop_id}")
    
    def get_incoming_link(self, to_node, loop_id):
        """Get the incoming link for a node on a specific loop"""
        # Find the link that ends at this node
        for key, link in self.links.items():
            if link.to_node == to_node and link.loop_id == loop_id:
                return link
        return None

class NodeInterface:
    """The routerless interface for a node"""
    def __init__(self, env, node_id, network, exb_capacity=5, num_ejection_links=2, debug=False):
        self.env = env
        self.node_id = node_id
        self.network = network
        
        # Determine loops passing through this node
        self.loops = self.get_loops_for_node()
        
        # Create buffers and resources
        self.loop_buffers = {loop_id: simpy.Store(env, capacity=1) for loop_id in self.loops}
        self.exb = simpy.Store(env, capacity=exb_capacity)  # Extension buffer
        self.ejection_links = [simpy.Resource(env, capacity=1) for _ in range(num_ejection_links)]
        self.ejection_queues = [simpy.Store(env) for _ in range(num_ejection_links)]
        
        # State tracking
        self.routing_table = {}  # Will be populated with {dest: best_loop_id}
        self.exb_attached_to_loop = {loop_id: False for loop_id in self.loops}
        self.reserved_ejection_link = None  # For livelock avoidance
        
        # Statistics
        self.injected_packets = 0
        self.ejected_packets = 0
        self.packet_latencies = []
        
        # Settings
        self.debug = debug
        
        # Start loop monitoring processes
        for loop_id in self.loops:
            # Monitor incoming links for each loop
            link = network.get_incoming_link(self.node_id, loop_id)
            if link:
                self.env.process(self.monitor_link(link))
    
    def get_loops_for_node(self):
        """Determine which loops pass through this node"""
        loops_for_node = []
        for loop_id, loop in self.network.loops.items():
            if self.node_id in loop.nodes:
                loops_for_node.append(loop_id)
        return loops_for_node
    
    def log(self, message):
        """Print debug logs if enabled"""
        if self.debug:
            print(f"Time {self.env.now}: Node {self.node_id} - {message}")
    
    def inject_packet(self, packet):
        """Inject a packet into the network"""
        self.injected_packets += 1
        yield self.env.timeout(0)  # Allow other processes to run first
        
        # Find the best loop for this destination
        best_loop_id = self.find_best_loop_for_destination(packet.dest)
        
        if best_loop_id is None:
            self.log(f"No loop available for destination {packet.dest}, packet {packet.id} dropped")
            return
        
        # Assign the loop to the packet
        packet.loop_id = best_loop_id
        
        # Break packet into flits
        flits = packet.flitation()
        self.log(f"=== INJECTING packet {packet.id} to destination {packet.dest} via loop {best_loop_id} ===")
        
        # If long packet, ensure an EXB is attached or available
        if len(flits) > 1:
            # Check if the loop is already attached to an EXB
            if not self.exb_attached_to_loop[best_loop_id]:
                # If not, check if EXB is full
                if len(self.exb.items) >= self.exb.capacity:
                    self.log(f"EXB FULL: Cannot inject packet {packet.id}, waiting")
                    # Wait for EXB to become available
                    yield self.env.timeout(1)
                    # Retry injection
                    yield self.env.process(self.inject_packet(packet))
                    return
                
                # Attach EXB to this loop
                self.exb_attached_to_loop[best_loop_id] = True
                self.log(f"Attaching EXB to loop {best_loop_id} for packet {packet.id}")
        
        # Inject flits one by one
        for flit in flits:
            # Check if loop is free
            if len(self.loop_buffers[best_loop_id].items) == 0:
                self.log(f"Injecting flit {flit.id} into loop {best_loop_id}")
                
                # Update the flit's current node
                flit.current_node = self.node_id
                
                # CRITICAL FIX: Set the actual injection time to current simulation time
                flit.injection_time = self.env.now
                
                # Put the flit in the loop buffer first
                yield self.loop_buffers[best_loop_id].put(flit)
                
                # Then process it to forward to the next node
                self.env.process(self.process_flit_on_loop(best_loop_id))
            else:
                # Loop is busy, store flit in EXB
                self.log(f"STORING IN EXB: Loop {best_loop_id} busy - flit {flit.id} goes to EXB")
                yield self.exb.put(flit)
            
            # Wait one cycle between injecting flits
            yield self.env.timeout(1)
        
        self.log(f"=== COMPLETED INJECTION of packet {packet.id} ===")
        
        # Drain EXB if anything stored there for this loop
        if self.exb_attached_to_loop[best_loop_id] and len(self.exb.items) > 0:
            yield self.env.process(self.drain_exb(best_loop_id))
    
    def drain_exb(self, loop_id):
        """Try to drain the EXB for a specific loop"""
        self.log(f"Attempting to drain EXB for loop {loop_id}")
        
        # Only drain while we have flits and the loop is still attached to EXB
        while len(self.exb.items) > 0 and self.exb_attached_to_loop[loop_id]:
            # Check if loop is available for draining
            if len(self.loop_buffers[loop_id].items) == 0:
                # Get a flit from EXB
                flit = yield self.exb.get()
                self.log(f"DRAINING FROM EXB: flit {flit.id} to loop {loop_id}")
                
                # Update flit's current node
                flit.current_node = self.node_id
                
                # CRITICAL FIX: Set the actual injection time to current simulation time
                flit.injection_time = self.env.now
                
                # Put the flit in the loop buffer
                yield self.loop_buffers[loop_id].put(flit)
                
                # Process the flit
                self.env.process(self.process_flit_on_loop(loop_id))
                
                # If EXB is empty, detach it from the loop
                if len(self.exb.items) == 0:
                    self.exb_attached_to_loop[loop_id] = False
                    self.log(f"Detaching EXB from loop {loop_id} - EXB is empty")
            else:
                self.log(f"Cannot drain EXB - loop {loop_id} is busy")
                break
            
            yield self.env.timeout(1)
    
    def monitor_link(self, link):
        """Monitor an incoming link for new flits"""
        while True:
            # Wait for a flit to arrive on this link
            flit = yield link.buffer.get()
            
            # Update flit's current node
            flit.current_node = self.node_id
            
            self.log(f"Received flit {flit.id} from loop {link.loop_id}")
            
            # Process the flit
            if len(self.loop_buffers[link.loop_id].items) == 0:
                # Put the flit in the loop buffer
                yield self.loop_buffers[link.loop_id].put(flit)
                
                # Process it
                self.env.process(self.process_flit_on_loop(link.loop_id))
            else:
                # This should not happen in a well-functioning network
                # as we should only have one flit per loop at a node
                self.log(f"ERROR: Loop {link.loop_id} buffer occupied when receiving flit {flit.id}")
    
    def process_flit_on_loop(self, loop_id):
        """Process a flit that has just entered the node on a loop"""
        # Get the flit from the loop buffer
        flit = yield self.loop_buffers[loop_id].get()
        
        self.log(f"Processing flit {flit.id} on loop {loop_id}")
        
        # Check if this is the destination
        if flit.dest == self.node_id:
            # Try to eject
            success = yield self.env.process(self.eject(flit))
            if not success:
                # If ejection failed (e.g., all ejection links busy), forward it
                yield self.env.process(self.forward_flit(flit))
        else:
            # Not for this node, forward it
            yield self.env.process(self.forward_flit(flit))
    
    def forward_flit(self, flit):
        """Forward a flit to the next node in its loop"""
        loop_id = flit.loop_id
        loop = self.network.loops[loop_id]
        
        # Get the next node in the loop
        next_node = loop.get_next_node(self.node_id)
        if next_node is None:
            self.log(f"ERROR: Cannot find next node for flit {flit.id} on loop {loop_id}")
            return
        
        self.log(f"Forwarding flit {flit.id} on loop {loop_id} to node {next_node}")
        
        # Transmit the flit to the next node
        yield self.network.transmit(self.node_id, next_node, loop_id, flit)
    
    def eject(self, flit):
        """Eject flit from the network"""
        # Check if we have a reserved ejection link for livelock avoidance
        if self.reserved_ejection_link is not None and flit.circling_count >= 254:
            ejection_link = self.ejection_links[self.reserved_ejection_link]
            
            with ejection_link.request() as req:
                yield req
                
                yield self.ejection_queues[self.reserved_ejection_link].put(flit)
                self.log(f"Ejecting flit {flit.id} via reserved ejection link {self.reserved_ejection_link}")
                
                # Record statistics for HEAD flits
                if flit.type == "HEAD":
                    flit.ejection_time = self.env.now
                    latency = flit.ejection_time - flit.injection_time
                    self.packet_latencies.append(latency)
                    self.log(f"LATENCY: Flit {flit.id} injected at {flit.injection_time}, ejected at {flit.ejection_time}, latency={latency}")

                
                if flit.type == "TAIL":
                    self.log(f"Completed ejection of packet {flit.pid}")
                    self.ejected_packets += 1
                    # Release the reserved link
                    self.reserved_ejection_link = None
                
                yield self.env.timeout(1)
                return True
        
        # Try each ejection link
        for link_idx, ejection_link in enumerate(self.ejection_links):
            # Check if link is available
            if ejection_link.count == 0:  # Resource is available
                with ejection_link.request() as req:
                    yield req
                    
                    yield self.ejection_queues[link_idx].put(flit)
                    self.log(f"Ejecting flit {flit.id} via ejection link {link_idx}")
                    
                    # Record statistics for HEAD flits
                    if flit.type == "HEAD":
                        flit.ejection_time = self.env.now
                        latency = flit.ejection_time - flit.injection_time
                        self.packet_latencies.append(latency)
                        self.log(f"LATENCY: Flit {flit.id} injected at {flit.injection_time}, ejected at {flit.ejection_time}, latency={latency}")

                    
                    if flit.type == "TAIL":
                        self.log(f"Completed ejection of packet {flit.pid}")
                        self.ejected_packets += 1
                    
                    yield self.env.timeout(1)
                    return True
        
        # All ejection links are busy, increment circling counter and return False
        if flit.type == "HEAD":
            flit.circling_count += 1
            self.log(f"No ejection links available - flit {flit.id} circling (count: {flit.circling_count})")
            
            # Livelock avoidance - if circling too many times, reserve an ejection link
            if flit.circling_count >= 254 and self.reserved_ejection_link is None:
                self.reserved_ejection_link = 0  # Reserve the first ejection link
                self.log(f"LIVELOCK AVOIDANCE - Marking ejection link 0 as reserved for flit {flit.id}")
        
        return False
    
    def find_best_loop_for_destination(self, dest):
        """Find the best loop to reach the destination"""
        # Check if we have a routing entry for this destination
        if dest in self.routing_table:
            return self.routing_table[dest]
        
        # Otherwise, find the loop with the shortest path to the destination
        best_loop_id = None
        min_hops = float('inf')
        
        for loop_id in self.loops:
            loop = self.network.loops[loop_id]
            path_length = loop.get_path_length(self.node_id, dest)
            
            if path_length < min_hops:
                min_hops = path_length
                best_loop_id = loop_id
        
        # Cache result in routing table
        if best_loop_id is not None:
            self.routing_table[dest] = best_loop_id
        
        return best_loop_id

class RoutelessNoC:
    """The complete Routerless NoC system"""
    def __init__(self, size, debug=False):
        self.size = size  # NxN size of the mesh
        self.env = simpy.Environment()
        self.debug = debug
        
        # Generate loops using the RLrec algorithm
        self.loops = self.generate_loops()
        
        # Create the physical network
        self.network = RoutelessNetwork(self.env, self.loops)
        
        # Create node interfaces
        self.nodes = self.create_nodes()
        
        # Statistics
        self.packet_counter = 0
        self.stats = {
            'injected_packets': 0,
            'ejected_packets': 0,
            'latencies': []
        }
        
        # Initialize random seed using current time
        self.seed = int(time_module.time())
    
    def generate_loops(self):
        """Generate loops using a simplified version of the RLrec algorithm from the paper"""
        loops = []
        n = self.size
        
        # For small NoCs, we can use predefined loop patterns
        if n == 2:
            # For 2x2, we use two simple loops (clockwise and counterclockwise)
            loop1_nodes = [0, 1, 3, 2, 0]  # Clockwise loop
            loop2_nodes = [0, 2, 3, 1, 0]  # Counterclockwise loop
            
            loops.append(Loop(0, loop1_nodes))
            loops.append(Loop(1, loop2_nodes))
            
        elif n == 4:
            # For 4x4, we use a combination of loops as described in the paper
            # Layer 1 (inner 2x2)
            inner_loop1 = [5, 6, 10, 9, 5]  # Clockwise
            inner_loop2 = [5, 9, 10, 6, 5]  # Counterclockwise
            
            # Layer 2 (outer loops)
            # Group A (single counterclockwise loop)
            outer_loop1 = [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 0]
            
            # Group B (first column as common edge)
            outer_loop2 = [0, 4, 8, 12, 13, 9, 5, 1, 0]
            outer_loop3 = [0, 4, 8, 12, 13, 14, 10, 6, 2, 1, 0]
            
            # Group C (last column as common edge)
            outer_loop4 = [3, 2, 6, 10, 14, 15, 11, 7, 3]
            outer_loop5 = [3, 2, 1, 5, 9, 13, 14, 15, 11, 7, 3]
            
            # Group D (horizontal "shortcut" loops)
            outer_loop6 = [0, 1, 2, 3, 0]
            outer_loop7 = [4, 5, 6, 7, 4]
            outer_loop8 = [8, 9, 10, 11, 8]
            outer_loop9 = [12, 13, 14, 15, 12]
            
            loops.append(Loop(0, inner_loop1))
            loops.append(Loop(1, inner_loop2))
            loops.append(Loop(2, outer_loop1))
            loops.append(Loop(3, outer_loop2))
            loops.append(Loop(4, outer_loop3))
            loops.append(Loop(5, outer_loop4))
            loops.append(Loop(6, outer_loop5))
            loops.append(Loop(7, outer_loop6))
            loops.append(Loop(8, outer_loop7))
            loops.append(Loop(9, outer_loop8))
            loops.append(Loop(10, outer_loop9))
        
        else:
            # For larger sizes, we'd implement the full RLrec algorithm
            # This is a simplified placeholder that creates a few loops
            
            # Create a perimeter loop clockwise
            perimeter_nodes = []
            # Top row left to right
            for i in range(n):
                perimeter_nodes.append(i)
            # Right column top to bottom (excluding corner)
            for i in range(1, n):
                perimeter_nodes.append(i*n + (n-1))
            # Bottom row right to left (excluding corner)
            for i in range(n-2, -1, -1):
                perimeter_nodes.append((n-1)*n + i)
            # Left column bottom to top (excluding corners)
            for i in range(n-2, 0, -1):
                perimeter_nodes.append(i*n)
            # Back to start
            perimeter_nodes.append(0)
            
            loops.append(Loop(0, perimeter_nodes))
            
            # Create a few more simpler loops
            # Horizontal rows
            for i in range(n):
                row_nodes = [i*n + j for j in range(n)]
                row_nodes.append(row_nodes[0])  # Back to start
                loops.append(Loop(i+1, row_nodes))
            
            # Vertical columns
            for i in range(n):
                col_nodes = [j*n + i for j in range(n)]
                col_nodes.append(col_nodes[0])  # Back to start
                loops.append(Loop(n+i+1, col_nodes))
        
        # Print the generated loops if debugging is enabled
        if self.debug:
            print("Generated Loops:")
            for loop in loops:
                print(loop)
        
        return loops
    
    def create_nodes(self):
        """Create node interfaces for each node in the NoC"""
        nodes = {}
        n = self.size
        num_nodes = n * n
        
        for node_id in range(num_nodes):
            nodes[node_id] = NodeInterface(
                env=self.env, 
                node_id=node_id,
                network=self.network,
                debug=self.debug
            )
        
        return nodes
    
    def generate_packet(self, src, dest=None, size=3, time=None):
        """Generate a new packet with a unique ID"""
        if time is None:
            time = self.env.now
        
        # If dest is not specified, choose a random destination
        if dest is None:
            n = self.size
            num_nodes = n * n
            dest = self.random_int(0, num_nodes - 1)
            # Make sure dest is not the same as src
            while dest == src:
                dest = self.random_int(0, num_nodes - 1)
        
        # Generate a unique packet ID
        packet_id = self.packet_counter
        self.packet_counter += 1
        
        return Packet(id=packet_id, src=src, dest=dest, size=size, time=time)
    
    def inject_packet(self, src, dest=None, size=3, time=None):
        """Inject a packet into the network from a source node"""
        packet = self.generate_packet(src, dest, size, time)
        self.stats['injected_packets'] += 1
        
        # Schedule the packet injection
        if time is not None and time > self.env.now:
            # Schedule for future
            delay = time - self.env.now
            self.env.process(self._delayed_injection(src, packet, delay))
        else:
            # Inject now
            self.env.process(self.nodes[src].inject_packet(packet))
        
        return packet
    
    def _delayed_injection(self, src, packet, delay):
        """Helper to inject a packet after a delay"""
        yield self.env.timeout(delay)
        yield self.env.process(self.nodes[src].inject_packet(packet))
    
    def run(self, until=100):
        """Run the simulation for a specified time"""
        # Reset statistics before each run
        self.stats = {
            'injected_packets': 0,
            'ejected_packets': 0,
            'latencies': []
        }
        
        self.env.run(until=until)
        
        # Gather statistics
        for node_id, node in self.nodes.items():
            self.stats['injected_packets'] += node.injected_packets
            self.stats['ejected_packets'] += node.ejected_packets
            self.stats['latencies'].extend(node.packet_latencies)
        
        # Calculate average latency
        if self.stats['latencies']:
            self.stats['avg_latency'] = sum(self.stats['latencies']) / len(self.stats['latencies'])
        else:
            self.stats['avg_latency'] = 0
        
        return self.stats
    
    # Simple random number generator functions to replace numpy/random
    def random_int(self, low, high):
        """Generate a random integer between low and high (inclusive)"""
        # Simple LCG random number generator
        self.seed = (1103515245 * self.seed + 12345) & 0x7fffffff
        return low + (self.seed % (high - low + 1))
    
    def random_float(self):
        """Generate a random float between 0 and 1"""
        self.seed = (1103515245 * self.seed + 12345) & 0x7fffffff
        return self.seed / 0x7fffffff
    
    def exponential(self, rate):
        """Generate a random sample from an exponential distribution"""
        # Using inverse transform sampling
        u = self.random_float()
        return -math.log(1.0 - u) / rate
    
    def generate_random_traffic(self, injection_rate, packet_size=3, duration=1000):
        """Generate random traffic with a specified injection rate"""
        n = self.size
        num_nodes = n * n
        
        # For each node, schedule packet injections
        for src in range(num_nodes):
            # Use Poisson process to model packet arrivals
            # Average packets per cycle = injection_rate
            # Inter-arrival time follows exponential distribution
            next_time = 0
            while next_time < duration:
                # Generate next arrival time
                inter_arrival = self.exponential(injection_rate)
                next_time += inter_arrival
                
                if next_time < duration:
                    # Random destination
                    dest = self.random_int(0, num_nodes - 1)
                    while dest == src:
                        dest = self.random_int(0, num_nodes - 1)
                    
                    # Inject packet
                    self.inject_packet(src, dest, packet_size, next_time)

# Example usage
                    
def run_noc_with_manual_packets(num_packets=20):
    # Create a smaller 2x2 NoC to increase congestion
    noc = RoutelessNoC(size=8, debug=True)  # Smaller network = more congestion
    
    # Total number of nodes in the network
    num_nodes = noc.size * noc.size  # 4 nodes for a 2x2 network
    
    # Inject packets with different source/destination pairs
    for i in range(num_packets):
        # Generate source node (0 to num_nodes-1)
        src = i % num_nodes
        
        # Generate destination node (different from source)
        dest = (src + 1) % num_nodes
        
        # Inject packets closer together to create congestion
        injection_time = i * 2  # Inject every 2 cycles (was 5)
        
        # Use larger packet sizes to increase network load
        packet_size = 3 + (i % 3)  # Sizes between 3-5 flits (was 2-4)
        
        # Inject the packet
        noc.inject_packet(
            src=src,
            dest=dest,
            size=packet_size,
            time=injection_time
        )
    
    # Run the simulation long enough for packets to reach destinations
    simulation_duration = (num_packets * 2) + 100  # Extra time for congestion
    stats = noc.run(until=simulation_duration)
    
    # Print statistics
    print("\nManual Traffic Simulation Statistics:")
    print(f"Injected packets: {stats['injected_packets']}")
    print(f"Ejected packets: {stats['ejected_packets']}")
    if stats['latencies']:
        print(f"Average packet latency: {stats['avg_latency']:.2f} cycles")
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
def run_routerless_noc_simulation():
    # Create a 4x4 Routerless NoC
    noc = RoutelessNoC(size=2, debug=True)
    
    # Inject some packets with time delays
    noc.inject_packet(src=0, dest=15, size=3, time=0)   # From node 0 to node 15
    noc.inject_packet(src=5, dest=10, size=2, time=10)  # From node 5 to node 10, 10 cycles later
    noc.inject_packet(src=12, dest=3, size=4, time=20)  # From node 12 to node 3, 20 cycles later
    
    # Run the simulation for longer to allow all packets to complete
    stats = noc.run(until=150)
    
    # Print statistics
    print("\nSimulation Statistics:")
    print(f"Injected packets: {stats['injected_packets']}")
    print(f"Ejected packets: {stats['ejected_packets']}")
    if stats['latencies']:
        print(f"Average packet latency: {stats['avg_latency']:.2f} cycles")
        print(f"Min latency: {min(stats['latencies'])} cycles")
        print(f"Max latency: {max(stats['latencies'])} cycles")

# Function to run a performance evaluation
def evaluate_performance(network_sizes=[2, 4], injection_rates=[0.01, 0.05, 0.1]):
    results = {}
    
    for size in network_sizes:
        results[size] = {'latency': {}, 'throughput': {}}
        
        for rate in injection_rates:
            # Run simulation with this injection rate
            noc = RoutelessNoC(size=size, debug=False)
            noc.generate_random_traffic(injection_rate=rate, duration=1000)
            stats = noc.run(until=1100)  # Run for slightly longer to finish processing
            
            # Record results
            results[size]['latency'][rate] = stats['avg_latency'] if stats['latencies'] else 0
            results[size]['throughput'][rate] = stats['ejected_packets'] / 1000  # Packets per cycle
    
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
    run_noc_with_manual_packets(20)
    
    # Uncomment to run a performance evaluation
    #evaluate_performance()