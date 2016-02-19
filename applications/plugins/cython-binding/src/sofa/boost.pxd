cdef extern from "boost/intrusive_ptr.hpp" namespace "boost":
        cdef cppclass intrusive_ptr[T]:
                intrusive_ptr()
                intrusive_ptr(T*)
                T* get()
