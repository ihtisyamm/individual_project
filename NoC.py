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
                src=self.src,
                dest=self.dest,
                time=self.timestamp
            ))
        return self.flits

class Flit:
    def __init__(self, id, pid, type, src, dest, time):
        self.id = id
        self.pid = pid # pid stands for packet id btw
        self.type = type # head | body | tail
        self.src = src
        self.dest = dest
        self.timestamp = time
