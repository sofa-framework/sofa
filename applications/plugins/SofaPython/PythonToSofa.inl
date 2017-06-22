#ifndef PYTHON_TO_SOFA_INL
#define PYTHON_TO_SOFA_INL

#include <SofaPython/PythonMacros.h>
#include <type_traits>

namespace sofa {
namespace py {

/// casting PyObject* to Sofa types
/// contains only inlined functions and must be included at the right place (!)
template<class T, class = void>
struct unwrap_traits {
    using wrapped_type = T;
    static const bool use_sptr = false;
};


namespace detail {
template<class T, class Base>
using requires_derived = typename std::enable_if< std::is_base_of<Base, T>::value>::type;
}

template<class T>
struct unwrap_traits<T, detail::requires_derived<T, sofa::core::objectmodel::Base> > {
    using wrapped_type = sofa::core::objectmodel::Base;
    static const bool use_sptr = true;
};





namespace detail {

/// unwrap a T* wrapped in a pyobject (via PySPtr, based on traits)
template<class T>
static inline typename std::enable_if< unwrap_traits<T>::use_sptr, T*>::type
unwrap(PyObject* obj, T*) {
    using wrapped_type = typename unwrap_traits<T>::wrapped_type;
    wrapped_type* ptr = ((PySPtr<wrapped_type>*)obj)->object.get();
    return dynamic_cast<T*>(ptr);
}


/// unwrap a T* wrapped in a pyobject (via PyPtr, based on traits)
template<class T>
static inline T* unwrap(PyObject* obj, ...) {
    using wrapped_type = typename unwrap_traits<T>::wrapped_type;
    wrapped_type* ptr = ((PyPtr<wrapped_type>*)obj)->object;
    
    return dynamic_cast<T*>(ptr);
}



/// wrap a T* in a pyobject (via PySPtr, based on traits)
template<class T>
static inline typename std::enable_if< unwrap_traits<T>::use_sptr, PyObject*>::type
wrap(T* obj, PyTypeObject *pto) {
    using wrapped_type = typename unwrap_traits<T>::wrapped_type;

    PySPtr<wrapped_type> * py_obj = (PySPtr<wrapped_type>*) PyType_GenericAlloc(pto, 0);
    py_obj->object = obj;
    return (PyObject*)py_obj;
}


/// unwrap a T* wrapped in a pyobject (via PyPtr, based on traits)
template<class T>
static inline typename std::enable_if< !unwrap_traits<T>::use_sptr, PyObject*>::type
wrap(T* obj, PyTypeObject *pto, bool deletable) {
    using wrapped_type = typename unwrap_traits<T>::wrapped_type;

    PyPtr<wrapped_type>* py_obj = (PyPtr<wrapped_type>*) PyType_GenericAlloc(pto, 0);
    py_obj->object = obj;
    py_obj->deletable = deletable;
    return (PyObject*)py_obj;
}

}


// unwrap a python pointer
template<class T>
static inline T* unwrap(PyObject* obj) {
    return detail::unwrap<T>(obj, 0);
}

// wrap a python object in a python object of type pto. you may need to pass
// extra boolean 'deletable' to specify python ownership (true = python may
// delete)
template<class T, class ... Args>
static inline PyObject* wrap(T* obj, PyTypeObject *pto, Args&& ... args) {
    return detail::wrap(obj, pto, std::forward<Args>(args)...);
}

}
}

#endif
