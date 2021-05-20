import socket
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
            lenrecv = b2int(tcp.recv(8))
            assert lensend == lenrecv, "Comm err"

    def recv(self, chost=None, cport=None):
        chost = chost or self.chost
        cport = cport or self.cport

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((chost, cport))
            sock.listen()
            tcp, client_address = sock.accept()
            # rk = b2int(ptcp.recv(8, socket.MSG_WAITALL))
            lenbytes = b2int(tcp.recv(8, fwait))
            argsbytes = tcp.recv(lenbytes, fwait)
            rarg, rkwargs = unpacking(argsbytes)
        return rarg, rkwargs

    def shutdown(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'shutdown'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)

    def read(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'read'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)

    """
        args : list of tuples; [args0, args1, args2, ...]
        kwargs : dict of dict;  {'0': kwargs0, '1': kwargs1, ...}
    """
    def read_parallel(self, *args, shost=None, sport=None, chost=None,
                      cport=None, nprc=None, **kwargs):
        nprc = nprc or self.nprc
        if args == ():
            args = [[]] * (nprc + 1)
        if kwargs == {}:
            kwargs = {'all': {}, **dict([("%d" % i, {}) for i in range(nprc)])}
        intype, instruction = b'1', b'read_parallel'
        self.send(*args[0], shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs['all'])

        chost = chost or self.chost
        cport = cport or self.cport

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((chost, cport))
            sock.listen()

            for i in range(nprc):
                tcp, client_address = sock.accept()
                rank = b2int(tcp.recv(8, fwait))
                tcp.send(packing(*args[rank+1], **kwargs[str(rank)]), fwait)

    def update(self, *args, shost=None, sport=None, **kwargs):
        intype, instruction = b'1', b'update'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)

    def update_parallel(self, *args, shost=None, sport=None, chost=None,
                        cport=None, nprc=None, **kwargs):
        nprc = nprc or self.nprc
        if args == ():
            args = [[]] * (nprc + 1)
        if kwargs == {}:
            kwargs = {'all': {}, **dict([("%d" % i, {}) for i in range(nprc)])}
        intype, instruction = b'1', b'update_parallel'
        self.send(*args[0], shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs['all'])

        chost = chost or self.chost
        cport = cport or self.cport

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((chost, cport))
            sock.listen()

            for i in range(nprc):
                tcp, client_address = sock.accept()
                rank = b2int(tcp.recv(8, fwait))
                tcp.send(packing(*args[rank+1], **kwargs[str(rank)]), fwait)

    def write(self, *args, shost=None, sport=None, chost=None, cport=None,
              **kwargs):
        intype, instruction = b'1', b'write'
        self.send(*args, shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs)
        rargs, rkwargs = self.recv(chost=chost, cport=cport)

        return rargs, rkwargs

    def write_parallel(self, *args, shost=None, sport=None, chost=None,
                       cport=None, nprc=None, **kwargs):

        nprc = nprc or self.nprc
        if args == ():
            args = [[]] * (nprc + 1)
        if kwargs == {}:
            kwargs = {'all': {}, **dict([("%d" % i, {}) for i in range(nprc)])}
        intype, instruction = b'1', b'write_parallel'
        self.send(*args[0], shost=shost, sport=sport, intype=intype,
                  instruction=instruction, **kwargs['all'])

        chost = chost or self.chost
        cport = cport or self.cport
        argsdct, kwargsdct = {}, {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((chost, cport))
            sock.listen()

            for i in range(nprc):
                tcp, client_address = sock.accept()
                rank = b2int(tcp.recv(8, fwait))
                tcp.send(packing(*args[rank+1], **kwargs[str(rank)]), fwait)

                lenbytes = b2int(tcp.recv(8, fwait))
                argsbytes = tcp.recv(lenbytes, fwait)
                pargs, pkwargs = unpacking(argsbytes)
                argsdct[rank] = pargs
                kwargsdct[rank] = pkwargs

        return argsdct, kwargsdct
