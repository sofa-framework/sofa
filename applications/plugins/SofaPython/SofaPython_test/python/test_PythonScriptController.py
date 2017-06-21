import Sofa

class TestController(Sofa.PythonScriptController):
    def __init__(self):
            return None

    def draw(self):
            print("HELLO")
            raise ValueError('A very specific bad thing happened')

