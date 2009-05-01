
from Network.Client import Client

__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$Apr 30, 2009 2:17:48 PM$"

class TCPClientAgent:
    """
    Simple wrapper for any agent to communicate TCPAgent on server side
    Connect to a server, receives observation, gives observation to the aggregated agent,
    takes action from aggregated agent and sends it back to the server.
    """
    client = None
    agent = None
    connected = False

    def __init__(self, Agent, host, port):
        """Documentation"""
        print "Initializing TCP Agent...";
        self.agent = Agent
        self.client = Client(host, port)
        self.connected = True
        pass

    def produceAction(self):
        """sends a string message"""
#        print "TCPAgent.produceAction: produsing Action..."
        a = self.getAction()
        self.client.sendData(a)
#        print "Action %s produced" % a

    def getObservation(self):
        """ receives and observation"""
#        print "Looking forward to receive data"
        data = self.client.recvData()
#        print "Data received: ", data
        if data == "-ciao":
            self.disconnet()
        elif len(data) > 100:
            self.agent.setObservation(data)
#        print data


    def getAction(self):
        return self.agent.getAction()

    def isConnected(self):
        """Ensures about connection to the server and let you know if """
        return self.connected