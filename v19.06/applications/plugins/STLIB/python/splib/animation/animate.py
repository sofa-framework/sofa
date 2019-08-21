# -*- coding: utf-8 -*-
import Sofa
from splib.utils import deprecated_alias

class Animation(object):
    """An animation clip that trigger callback at regular intervales for a given duration.

           :param mode: specify how the animation will continue (None, "loop", "pingpong")
           :param duration: the duration of the animation in seconds.
           :param onUpdate: callback function called each update.
           :param onDone: callback function called when the animation is terminated.
           :param params: a dictionnary with user specified extra parameters that are passed to the callback.

    Example of use:


    .. code-block:: python

        def createScene(rootNode):
            AnimationManager(rootNode)

            def onUpdate(target, factor):
                print("Callback called on: "+target.name+" factor is: "+str(factor))

            def onDone(target, factor):
                print("Callback when animation is done: "+target.name+" factor is: "+str(factor))

            animate(onUpdate, {"target" : rootNode }, 12, onDone=onDone)
    """
    @deprecated_alias(cb='onUpdate')
    def __init__(self, duration, mode, onUpdate, params, onDone=None):
        if 'startTime' in params:
            self.startTime = params['startTime']
        else:
            self.startTime = None

        self.duration = duration
        self.onUpdate = onUpdate        
        self.onDone = onDone
        self.params = params
        self.factor = 1.0
        self.direction = 1.0
        self.mode = mode

    def doOnDone(self, currentTime):
        self.onDone(factor=self.factor, **self.params)

    def update(self, currentTime):
        if self.startTime is None:
            self.startTime = currentTime

        if self.duration == 0.0:
            self.factor = 1.0
        elif self.direction > 0.0:
            self.factor = (currentTime-self.startTime) / self.duration
        else:
            self.factor = 1.0-(currentTime-self.startTime) / self.duration

        if self.factor > 1.0:
            self.factor = 1.0

        if self.factor < 0.0:
            self.factor = 0.0

        self.onUpdate(factor=self.factor, **self.params)


class AnimationManagerController(Sofa.PythonScriptController):
    """Implements the AnimationManager as a PythonScriptController
    """
    def __init__(self, node):
        self.listening = True
        self.name = "AnimationManager"
        self.totalTime = 0
        self.animations = []

    def addAnimation(self, animation):
        self.animations.append(animation)

    def removeAnimation(self, animation):
        self.animations.remove(animation)

    def bwdInitGraph(self, root):
        self.onBeginAnimationStep(0.0)

    def onBeginAnimationStep(self, dt):
        self.totalTime += dt
        nextanimations = []
        for animation in self.animations:
            animation.update(self.totalTime)
            if animation.factor < 1.0 and animation.direction > 0.0:
                nextanimations.append(animation)
            elif animation.factor > 0.0 and animation.direction < 0.0:
                nextanimations.append(animation)
            elif animation.mode == "pingpong":
                animation.direction = -animation.direction
                animation.startTime = None
                nextanimations.append(animation)
            elif animation.mode == "loop":
                animation.direction = animation.direction
                animation.startTime = None
                nextanimations.append(animation)
            elif animation.onDone is not None:
                animation.doOnDone(self.totalTime)
        self.animations = nextanimations
        return 0


manager = None

def animate(onUpdate, params, duration, mode="once", onDone=None):
    """Construct and starts an animation

    Build a new animation from a callback function that computes the animation value,
    a set of parameters, the animation duration and the type of animation repetition pattern.

    Animation can be added from any code location (createScene, PythonScriptController)

    :param float duration: duration of the animation in seconds.
    :param str mode: once, loop, pingpong 

    Example:
        .. sourcecode:: python

            def myAnimate(target, factor):
                print("I should do something on: "+target.name)


            def createScene(rootNode)
                AnimationManager(rootNode)
                animate(myAnimate, {"target" : rootNode }, 10)
    """
    if manager == None:
        raise Exception("Missing manager in this scene")
        
    manager.addAnimation(Animation(duration=duration, mode=mode, onUpdate=onUpdate, params=params, onDone=onDone)) 


def removeAnimation(animation):
    if manager == None:
        raise Exception("Missing manager in this scene")
        
    manager.removeAnimation(animation) 

def AnimationManager(node):
    """
    A Controller to manage all animations in the scene

    Before using the animation framework an AnimationManager
    must be added to the scene. It has in charge, at each time step
    to update all the running animations.

    Returns:
        AnimationManagerController

    Example:
        .. sourcecode:: python

            def createScene(rootNode)
                AnimationManager(rootNode)
    """
    global manager
    if manager is not None:
        Sofa.msg_info(node, "There is already one animation manager in this scene...why do you need a second one ?") 
        return manager
    manager = AnimationManagerController(node)
    return manager


# This function is just an example on how to use the animate function.
def createScene(rootNode):
    def myAnimate1(target, factor):
        print("I should do something on: "+target.name+" factor is: "+str(factor))

    def myAnimate2(target, factor):
        print("Function 2: "+target.name+" factor is: "+str(factor))

    def myOnDone(target, factor):
        print("onDone: "+target.name+" factor is: "+str(factor))

    AnimationManager(rootNode)
    animate(myAnimate1, {"target": rootNode}, 10)
    animate(myAnimate2, {"target": rootNode}, 12, onDone=myOnDone)
