import simpy
import random

def packet_generator(env, source, destination, injection_rate, packet_size, network):
    """Generate packets according to injection rate"""
    while True:
        # Wait before generating next packet
        yield env.timeout(random.expovariate(injection_rate))
        
        # Create packet
        packet_id = f"packet_{source}_{destination}_{env.now:.2f}"
        packet = {
            'id': packet_id,
            'source': source,
            'destination': destination,
            'size': packet_size,  # in flits
            'creation_time': env.now,
            'current_pos': source
        }
        
        print(f"Time {env.now:.2f}: Generated {packet_id} from {source} to {destination}")
        
        # Start packet transmission process
        env.process(packet_transmission(env, packet, network))

def packet_transmission(env, packet, network):
    """Handle packet movement through the ring"""
    current_pos = packet['source']
    
    # Find next node in ring towards destination
    while current_pos != packet['destination']:
        # Get next node in ring
        ring = network['ring']
        current_idx = ring.index(current_pos)
        next_idx = (current_idx + 1) % len(ring)
        next_pos = ring[next_idx]
        
        # Simulate flit-by-flit transmission
        for flit in range(packet['size']):
            # Wait one cycle for flit transmission
            yield env.timeout(1)
            
            # Update packet position
            packet['current_pos'] = next_pos
            
            if flit == 0:  # First flit (header)
                print(f"Time {env.now:.2f}: {packet['id']} header reached {next_pos}")
            elif flit == packet['size'] - 1:  # Last flit
                print(f"Time {env.now:.2f}: {packet['id']} fully arrived at {next_pos}")
        
        current_pos = next_pos
        
        # If reached destination, calculate and record latency
        if current_pos == packet['destination']:
            latency = env.now - packet['creation_time']
            network['latencies'].append(latency)
            print(f"Time {env.now:.2f}: {packet['id']} completed with latency {latency:.2f}")

def run_simulation():
    # Simulation parameters
    SIM_TIME = 100      # Simulation time
    PACKET_SIZE = 4     # Flits per packet
    INJECTION_RATE = 0.1  # Packets per time unit
    
    # Create SimPy environment
    env = simpy.Environment()
    
    # Define simple ring topology (4 nodes)
    network = {
        'ring': ['node0', 'node1', 'node2', 'node3'],
        'latencies': []  # Store packet latencies
    }
    
    # Create packet generators for each node
    for source in network['ring']:
        for destination in network['ring']:
            if source != destination:  # Don't send packets to self
                env.process(packet_generator(
                    env, 
                    source, 
                    destination, 
                    INJECTION_RATE,
                    PACKET_SIZE,
                    network
                ))
    
    # Run simulation
    env.run(until=SIM_TIME)
    
    # Print statistics
    if network['latencies']:
        avg_latency = sum(network['latencies']) / len(network['latencies'])
        print(f"\nSimulation completed:")
        print(f"Average packet latency: {avg_latency:.2f}")
        print(f"Total packets delivered: {len(network['latencies'])}")

if __name__ == "__main__":
    run_simulation()