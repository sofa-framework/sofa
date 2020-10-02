
try:
    import numpy
    import SofaPython.SofaNumpy

    def image_as_numpy( img, index=-1 ):
        '''maps image content as a list of numpy arrays with shared memory'''

        ptrs, shape, typename = img.getPtrs()

        type = SofaPython.SofaNumpy.ctypeFromName.get(typename,None)
        if not type: raise Exception("can't map image of type " + typename)


        if index ==-1:

            imglist = []

            for p in ptrs:
                array = SofaPython.SofaNumpy.ctypes.cast( SofaPython.SofaNumpy.ctypes.c_void_p(p), SofaPython.SofaNumpy.ctypes.POINTER(type))
                imglist.append(numpy.ctypeslib.as_array( array, shape ))

            return imglist

        else:
            # assert( index < len(ptrs) )
            array = SofaPython.SofaNumpy.ctypes.cast( SofaPython.SofaNumpy.ctypes.c_void_p(ptrs[index]), SofaPython.SofaNumpy.ctypes.POINTER(type))
            return numpy.ctypeslib.as_array( array, shape )

except ImportError: # numpy is not mandatory
    pass


