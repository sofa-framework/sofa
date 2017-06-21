import Sofa

def f3():
    raise ValueError('The value is not valid')

class TestController(Sofa.PythonScriptController):
    def __init__(self):
            return None

    def onLoaded(self, node):
        print("LOADED !")
        name = findData("name")
        name.setValue( 123 )

    def f2(self):
        f3()

    def draw(self):
        self.f2()
