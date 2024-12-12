import collections
from typing import Dict, List, Tuple

# Constants
FLIT_SIZE = 1  # Size of each flit in bytes
MAX_PACKET_SIZE = 16  # Maximum packet size in flits
SIMULATION_CYCLES = 1000  # Number of cycles to simulate

# Basic data structures
injection_queue = collections.deque()  # Queue for packets waiting to be injected
flit_buffer = collections.deque(maxlen=1)  # Single flit buffer for incoming flits
packet_buffer = collections.deque()  # Buffer for storing complete packets
current_cycle = 0  # Current simulation cycle

# Track statistics
total_packets_sent = 0
total_packets_received = 0
packet_latencies = []

def create_packet(source: int, destination: int, size: int, creation_time: int) -> Dict:
    """Create a new packet with header and payload flits"""
    return {
        'source': source,
        'destination': destination,
        'size': size,
        'creation_time': creation_time,
        'flits_remaining': size,
        'header_sent': False
    }

def inject_packet(packet: Dict) -> bool:
    """Try to inject a packet into the network"""
    global total_packets_sent
    # Can only inject if flit buffer is empty
    if not flit_buffer and not packet['header_sent']:
        # Inject header flit first
        flit_buffer.append(('header', packet))
        packet['header_sent'] = True
        packet['flits_remaining'] -= 1
        total_packets_sent += 1
        print(f"Cycle {current_cycle}: Injected packet from {packet['source']} to {packet['destination']}")
        return True
    return False

def process_flit_buffer():
    """Process flits in the buffer"""
    if flit_buffer:
        flit_type, packet = flit_buffer.popleft()
        # If this is our destination, receive the packet
        if packet['destination'] == our_core_id:
            receive_packet(packet)
        else:
            # Forward to next switch in ring
            packet['flits_remaining'] -= 1
            print(f"Cycle {current_cycle}: Forwarded flit for packet to destination {packet['destination']}")

def receive_packet(packet: Dict):
    """Handle received packet"""
    global total_packets_received
    total_packets_received += 1
    latency = current_cycle - packet['creation_time']
    packet_latencies.append(latency)
    print(f"Cycle {current_cycle}: Received packet from {packet['source']}, latency: {latency}")

def simulate_cycle():
    """Simulate one cycle of the network"""
    global current_cycle
    
    # Try to inject waiting packets
    if injection_queue:
        packet = injection_queue[0]
        if inject_packet(packet):
            if packet['flits_remaining'] == 0:
                injection_queue.popleft()
    
    # Process flits in buffer
    process_flit_buffer()
    
    current_cycle += 1

def run_simulation():
    """Run the main simulation loop"""
    global current_cycle
    
    # Add test packets to injection queue - now sending to our own core (0)
    injection_queue.append(create_packet(0, 0, 5, current_cycle))
    injection_queue.append(create_packet(0, 0, 3, current_cycle))
    
    # Run simulation for specified number of cycles
    while current_cycle < SIMULATION_CYCLES:
        simulate_cycle()
        
        # Add more packets based on some traffic pattern
        if current_cycle % 10 == 0:  # Add packet every 10 cycles
            injection_queue.append(create_packet(0, 0, 5, current_cycle))
    
    # Print statistics
    print("\nSimulation Results:")
    print(f"Total packets sent: {total_packets_sent}")
    print(f"Total packets received: {total_packets_received}")
    if packet_latencies:
        avg_latency = sum(packet_latencies) / len(packet_latencies)
        print(f"Average packet latency: {avg_latency} cycles")

# Set core ID
our_core_id = 0

if __name__ == "__main__":
    run_simulation()