import Sofa

# this eases the process of adding a script and getting its handle


event_name = 'RetrieveHandle'

class Base(Sofa.PythonScriptController):

    def onScriptEvent(self, sender, event, data):
        # print 'retrieving script handle'

        if event == event_name:
            data.res = self;

        return 0


def insert( node, **kwargs ):
    
    node.createObject('PythonScriptController',
                      **kwargs)
    # TODO: check that classname value derives from Base

    class Data:
        pass

    data = Data()

    node.sendScriptEvent(event_name, data)

    # TODO clean error message if data.res was not found 

    return data.res

