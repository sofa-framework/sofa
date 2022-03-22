import Sofa
import Sofa.Core

import tkinter as tkinter
import threading
import numpy
from math import pi


class App(threading.Thread):

    def __init__(self, initAngles=[0.,0.,0.,0.,0.,0.]):
        threading.Thread.__init__(self)
        self.daemon = True
        self.start()
        self.angle1Init = initAngles[0]
        self.angle2Init = initAngles[1]
        self.angle3Init = initAngles[2]
        self.angle4Init = initAngles[3]
        self.angle5Init = initAngles[4]
        self.angle6Init = initAngles[5]

    def reset(self):
        self.angle1.set(self.angle1Init)
        self.angle2.set(self.angle2Init)
        self.angle3.set(self.angle3Init)
        self.angle4.set(self.angle4Init)
        self.angle5.set(self.angle5Init)
        self.angle6.set(self.angle6Init)

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = tkinter.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        tkinter.Label(self.root, text="Robot Controller Interface").grid(row=0, columnspan=6)

        self.angle1 = tkinter.DoubleVar()
        self.angle2 = tkinter.DoubleVar()
        self.angle3 = tkinter.DoubleVar()
        self.angle4 = tkinter.DoubleVar()
        self.angle5 = tkinter.DoubleVar()
        self.angle6 = tkinter.DoubleVar()

        tkinter.Scale(self.root, variable=self.angle1, resolution=0.001, length=400, from_=-pi, to=pi, orient=tkinter.VERTICAL).grid(row=1, column=0)
        tkinter.Scale(self.root, variable=self.angle2, resolution=0.001, length=400, from_=-pi, to=pi, orient=tkinter.VERTICAL).grid(row=1, column=1)
        tkinter.Scale(self.root, variable=self.angle3, resolution=0.001, length=400, from_=-pi, to=pi, orient=tkinter.VERTICAL).grid(row=1, column=2)
        tkinter.Scale(self.root, variable=self.angle4, resolution=0.001, length=400, from_=-pi, to=pi, orient=tkinter.VERTICAL).grid(row=1, column=3)
        tkinter.Scale(self.root, variable=self.angle5, resolution=0.001, length=400, from_=-pi, to=pi, orient=tkinter.VERTICAL).grid(row=1, column=4)
        tkinter.Scale(self.root, variable=self.angle6, resolution=0.001, length=400, from_=-pi, to=pi, orient=tkinter.VERTICAL).grid(row=1, column=5)

        self.root.mainloop()


class RobotGUI(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self,args,kwargs)
        self.robot = kwargs["robot"]
        self.app = App(kwargs.get("initAngles",[0.,0.,0.,0.,0.,0.]))

        return

    def reset(self):
        self.app.reset()

    def onAnimateBeginEvent(self, event):

        if self.robot is None:
            return

        angles = [
                self.app.angle1.get(),
                self.app.angle2.get(),
                self.app.angle3.get(),
                self.app.angle4.get(),
                self.app.angle5.get(),
                self.app.angle6.get()
                ]

        angles = numpy.array(angles)
        self.robot.angles = angles.tolist()

        return


# Test/example scene
def createScene(rootNode):

    from header import addHeader

    addHeader(rootNode)
    rootNode.addObject(RobotGUI(robot=None))

    return
