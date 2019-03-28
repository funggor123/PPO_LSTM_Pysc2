import configparser
from dist.node import Node


def readConfig(config_file_name):
    config = configparser.ConfigParser()
    config.read(config_file_name)

    worker = []

    worker_section = config.items("Worker_Section")
    for item in worker_section:
        worker.append(Node(port=item[0], ip=item[1]).get_port_ip())

    ps = []

    ps_section = config.items("Ps_Section")
    for item in ps_section:
        ps.append(Node(port=item[0], ip=item[1], type="ps").get_port_ip())

    print("[Track] Number of Ps : " + str(len(ps)))
    print("[Track] Number of Worker : " + str(len(worker)))
    return worker, ps