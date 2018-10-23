import numpy

class RGBAColor(numpy.ndarray):
    """A wrapping-type to manipulate Sofa data field as an RGBAColor.

       Eg:
           n = Sofa.Node("root")
           m = n.addObject("MechanicalObject")
           c = RGBAColor(m.showColor)
           print("Red is: ", c.r())
    """

    def __new__(cls, input_array=None):
           import Sofa
           if input_array is None:
               obj = super(RGBAColor, cls).__new__(cls, shape=(4), dtype=float)               
               obj[0] = obj[1] = obj[2] = obj[3] = 0 
               return obj

           if isinstance(input_array, list):
               if len(input_array) not in [4]:
                   raise ValueError("The list is too long. Only size 4 list is supported")

               obj = super(RGBAColor, cls).__new__(cls, shape=(4), dtype=float)
               numpy.copyto(obj, numpy.asarray(input_array))
               return obj
               
           if isinstance(input_array, Sofa.Core.DataContainer):
               cls.owner = input_array
               input_array = input_array.toarray()

           if input_array.ndim != 1:
               raise ValueError("Invalid dimension, expecting a 1D array, got "+str(input_array.ndim)+"D")

           # Input array is an already formed ndarray instance
           # We first cast to be our class type
           obj = numpy.asarray(input_array).view(cls)

           # Finally, we must return the newly created object:
           return obj

    def r(self):
        return self[0]

    def g(self):
        return self[1]

    def b(self):
        return self[2]

    def a(self):
        return self[3]
