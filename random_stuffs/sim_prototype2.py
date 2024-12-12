import simpy
import collections
from typing import Dict, List, Tuple

class RouterslessNoC:
    def __init__(self, env: simpy.Environment):
        # SimPy environment
        self.env = env
        
        # Network components
        self.flit_buffer = collections.deque(maxlen=1)  # Single flit buffer
        self.packet_buffer = collections.deque()  # Packet buffer
        self.injection_queue = collections.deque()  # Injection queue
        
        # Statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.latencies = []
        
        # Start the switch process
        self.env.process(self.switch_cycle())

    def create_packet(self, source: int, dest: int, size: int) -> Dict:
        """Create a new packet"""
        return {
            'id': self.packets_sent,
            'source': source,
            'dest': dest,
            'size': size,
            'flits_remaining': size,
            'header_sent': False,
            'creation_time': self.env.now
        }

    def inject_packet(self, packet: Dict) -> bool:
        """Try to inject a packet into the network"""
        # Can only inject if flit buffer is empty
        if not self.flit_buffer and not packet['header_sent']:
            # Inject header flit
            self.flit_buffer.append(('header', packet))
            packet['header_sent'] = True
            packet['flits_remaining'] -= 1
            self.packets_sent += 1
            print(f"Time {self.env.now}: Injected packet {packet['id']} from {packet['source']} to {packet['dest']}")
            return True
        return False

    def process_flit_buffer(self):
        """Process flits in the buffer"""
        if self.flit_buffer:
            flit_type, packet = self.flit_buffer.popleft()
            if packet['dest'] == 0:  # Assuming we're core 0
                self.receive_packet(packet)
            else:
                # Forward to next switch in ring
                packet['flits_remaining'] -= 1
                print(f"Time {self.env.now}: Forwarded flit for packet {packet['id']}")

    def receive_packet(self, packet: Dict):
        """Handle received packet"""
        self.packets_received += 1
        latency = self.env.now - packet['creation_time']
        self.latencies.append(latency)
        print(f"Time {self.env.now}: Received packet {packet['id']}, latency: {latency}")

    def switch_cycle(self):
        """Main switch operation process"""
        while True:
            # Try to inject waiting packets
            if self.injection_queue:
                packet = self.injection_queue[0]
                if self.inject_packet(packet):
                    if packet['flits_remaining'] == 0:
                        self.injection_queue.popleft()

            # Process flits in buffer
            self.process_flit_buffer()

            # Wait for next cycle
            yield self.env.timeout(1)

def traffic_generator(env: simpy.Environment, noc: RouterslessNoC):
    """Generate traffic at regular intervals"""
    while True:
        # Create and queue new packet
        packet = noc.create_packet(0, 0, 5)  # Source 0, Dest 0, Size 5
        noc.injection_queue.append(packet)
        
        # Wait before generating next packet
        yield env.timeout(10)  # Generate packet every 10 cycles

def run_simulation(duration: int = 100):
    """Run the simulation"""
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create NoC
    noc = RouterslessNoC(env)
    
    # Start traffic generator
    env.process(traffic_generator(env, noc))
    
    # Run simulation
    env.run(until=duration)
    
    # Print statistics
    print("\nSimulation Results:")
    print(f"Total packets sent: {noc.packets_sent}")
    print(f"Total packets received: {noc.packets_received}")
    if noc.latencies:
        avg_latency = sum(noc.latencies) / len(noc.latencies)
        print(f"Average packet latency: {avg_latency} cycles")

if __name__ == "__main__":
    run_simulation()