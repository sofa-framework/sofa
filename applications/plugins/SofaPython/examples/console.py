def createScene(node):

    def hello():
        print 'hello !'

    print "usage: as usual, call dir() to see what's available."
    
    from SofaPython import console # the console is started during the import
    c = console.Console( locals() )
    

