class Node:

    def __init__(self, port=2222, ip="127.0.0.1", type="worker"):
        self._port = port
        self._ip = ip
        self._type = type
        print("[Track] Node: " + self.get_ip() + ":" + str(self.get_port()) + " " + type)

    def get_ip(self):
        return self._ip

    def get_port(self):
        return self._port

    def get_port_ip(self):
        return self.get_ip() + ":" + str(self.get_port())
