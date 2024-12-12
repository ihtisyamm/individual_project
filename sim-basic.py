import simpy

class Switch:
    def __init__(self, env, inputStore) -> None:
        self.env = env
        self.buffer = simpy.Store(env)
        self.output = simpy.Store(env)
        self.inputStore = inputStore

        self.env.process(self.inputToBuffer())
        self.env.process(self.bufferToOutput())

    def inputToBuffer(self):
        while True:
            if self.inputStore:
                data = yield self.inputStore.get()
                yield self.buffer.put(data)
            yield self.env.timeout(1)

    def bufferToOutput(self):
        while True:
            if self.buffer.items:
                data = yield self.buffer.get()
                yield self.output.put(data)
            yield self.env.timeout(1)

    def getBufferOutput(self):
        # return buffer and output
        return self.buffer, self.output
    
def input_process(env, output, list):
    for data in list:
        yield env.timeout(1)
        yield output.put(data)

def receiver(env, stored):
    while True:
        item = yield stored.get()
        print(f"Time {env.now}: Output received {item}")
        yield env.timeout(1)

def main():
    env = simpy.Environment()

    input_store = simpy.Store(env)

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8]

    switch = Switch(env, input_store)
    buffer, output = switch.getBufferOutput()

    env.process(input_process(env, input_store, data))
    env.process(receiver(env, output))

    env.run(until=20)

if __name__ == "__main__":
    main()