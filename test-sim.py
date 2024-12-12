import simpy

def producer(env, store):
    for i in range(5):
        yield env.timeout(1)
        yield store.put(i)  # Will wait if store is full
        print(f"Added {i} to store")

def consumer(env, store):
    yield env.timeout(4)  # Wait a bit to let store fill up
    while True:
        item = yield store.get()
        print(f"Got {item} from store")
        yield env.timeout(1)

env = simpy.Environment()
store = simpy.Store(env, capacity=3)  # Can only hold 3 items
env.process(producer(env, store))
env.process(consumer(env, store))
env.run()