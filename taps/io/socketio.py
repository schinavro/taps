import numpy as np
import socket
import time
from socket import MSG_WAITALL as fwait
from taps.utils.antenna import packing, unpacking


def int2b(integer): return int.to_bytes(integer, 8, 'little')


def b2int(bytes): return int.from_bytes(bytes, 'little', signed=True)


class SocketIO:
    """
    chost : client host;
    cport : client port;
    shost : server host;
    sport : server port;
    """
    def __init__(self, chost="127.0.0.1", cport=6544,
                 shost="127.0.0.1", sport=6543, nprc=1):
        self.chost = chost
        self.cport = cport
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((chost, cport))
        self.sock.listen()
        self.shost = shost
        self.sport = sport
        self.nprc = nprc

    def send(self, *args, shost=None, sport=None, intype=b'1',
             instruction=None, **kwargs):
        shost = shost or self.shost
        sport = sport or self.sport
        argsbytes = packing(*args, **kwargs)
        sendbytes = intype + int2b(len(instruction)) + instruction + argsbytes
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            tcp.connect((shost, sport))
            lensend = len(sendbytes)
            tcp.send(int2b(lensend) + sendbytes)
            lenrecv = b2int(tcp.recv(8, fwait))
            assert lensend == lenrecv, "Comm err"

    def recv(self):

        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            #sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            #sock.bind((chost, cport))
            #sock.listen()
        tcp, client_address = self.sock.accept()
        lenbytes = b2int(tcp.recv(8, fwait))
        argsbytes = tcp.recv(lenbytes, fwait)
        rarg, rkwargs = unpacking(argsbytes)
        return rarg, rkwargs

    def ping(self, shost=None, sport=None):
        intype, instruction = b'1', b'ping'
        shost = shost or self.shost
        sport = sport or self.sport
        argsbytes = packing()
        sendbytes = intype + int2b(len(instruction)) + instruction + argsbytes
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
            # Ping
            a = time.time()
            tcp.connect((shost, sport))
            lensend = len(sendbytes)
            tcp.send(int2b(lensend) + sendbytes)
            lenrecv = b2int(tcp.recv(8, fwait))
            assert lensend == lenrecv, "Comm err"
            # Pong
            b = np.frombuffer(tcp.recv(8, fwait), dtype=np.float64)[0]
            print("%.2f" % ((b - a) * 1000), "ms")

    def shutdown(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'shutdown'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)
        self.sock.close()

    def read(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'read'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)

    def read_parallel(self, *args, shost=None, sport=None, nprc=None, **kwargs):
        """
            args : list of tuples; [args0, args1, args2, ...]
            kwargs : dict of dict;  {'all': {}, 'rank0': kwargs0, 'rank1': ...}
        """
        nprc = nprc or self.nprc
        if args == ():
            args = [[]] * (nprc + 1)
        if kwargs == {}:
            kwargs = {'all': {}, **dict([("rank%d" % i, {}) for i in range(nprc)])}
        intype, instruction = b'1', b'read_parallel'
        self.send(*args[0], shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs['all'])

        # chost = chost or self.chost
        # cport = cport or self.cport

        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        #    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #    sock.bind((chost, cport))
        #    sock.listen()
        try:
            for i in range(nprc):
                print("Accept", i)
                tcp, client_address = self.sock.accept()
                print("Recv", i)
                rank = b2int(tcp.recv(8, fwait))
                print("Send", i)
                tcp.send(packing(*args[rank+1], **kwargs['rank'+str(rank)]))
        except KeyboardInterrupt as e:
            self.sock.close()
            raise e

    def update(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'update'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)

    def update_parallel(self, *args, shost=None, sport=None, nprc=None,
                        **kwargs):
        nprc = nprc or self.nprc
        if args == ():
            args = [[]] * (nprc + 1)
        if kwargs == {}:
            kwargs = {'all': {}, **dict([("rank%d" % i, {}) for i in range(nprc)])}
        intype, instruction = b'1', b'update_parallel'
        self.send(*args[0], shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs['all'])

        for i in range(nprc):
            tcp, client_address = self.sock.accept()
            rank = b2int(tcp.recv(8, fwait))
            tcp.send(packing(*args[rank+1], **kwargs['rank'+str(rank)]))

    def write(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'write'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)
        rargs, rkwargs = self.recv()

        return rargs, rkwargs

    def write_parallel(self, *args, shost=None, sport=None, nprc=None,
                       **kwargs):

        nprc = nprc or self.nprc
        if args == ():
            args = [[]] * (nprc + 1)
        if kwargs == {}:
            kwargs = {'all': {}, **dict([("rank%d" % i, {}) for i in range(nprc)])}
        intype, instruction = b'1', b'write_parallel'
        self.send(*args[0], shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs['all'])

        argsdct, kwargsdct = {}, {}
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #     sock.bind((chost, cport))
        #     sock.listen()

        for i in range(nprc):
            tcp, client_address = self.sock.accept()
            rank = b2int(tcp.recv(8, fwait))
            tcp.send(packing(*args[rank+1],
                             **kwargs['rank'+str(rank)]))

            lenbytes = b2int(tcp.recv(8, fwait))
            argsbytes = tcp.recv(lenbytes, fwait)
            pargs, pkwargs = unpacking(argsbytes)
            argsdct[rank] = pargs
            kwargsdct[rank] = pkwargs

        return argsdct, kwargsdct
