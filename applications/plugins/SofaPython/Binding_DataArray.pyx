from cython cimport view
from libcpp cimport bool

cdef public makeTypedArray():
        my_array = view.array(shape=(100, 7), itemsize=sizeof(double), format="d")       
        return my_array
        
cdef public Data_toList(o):
    if o.ndim == 0:
        return o.value
    elif o.ndim == 1:
        tmp=[]
        for i in range(o.shape[0]):
            tmp.append(o[i])

        return tmp
        
    elif o.ndim == 2:
        tmp=[]
        for i in range(o.shape[0]):
            tmp2=[]
            for j in range(o.shape[1]):
                tmp2.append(o[i,j])
            tmp.append(tmp2)
        return tmp
    else:
        return None

def Data_helper(o, key, value):
        o[key] = value

def sliceToRange(c, end):
        start = 0
        stop = end
        if c.start != None:
                start = c.start
        if c.stop != None:
                stop = c.stop
        return xrange(start, stop) 
        
cdef public bool Data_ass_subscript(o,key,value):
        if isinstance(key, slice):
                start, stop, step = key.start, key.stop, key.step
                for i in xrange(start,stop):
                        Data_helper(o, i, value)
                return True
        elif callable(key):
                g = key(o)        
                for i in g:
                        Data_helper(o, i, value)
                return True
        elif isinstance(key, tuple):
                if isinstance(key[0], slice) and isinstance(key[1], slice):
                        igen =  sliceToRange(key[0], o.shape[0])
                        jgen =  sliceToRange(key[1], o.shape[1])
                        for i in igen:
                                for j in jgen:
                                        Data_helper(o,(i,j),value)                     
                        return True                        
        return False
