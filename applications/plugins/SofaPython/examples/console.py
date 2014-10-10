from SofaPython import console


def createScene(node):

    def hello():
        print 'hello !'

    print "usage: as usual, call dir() to see what's available."
    c = console.Console( locals() )
    

