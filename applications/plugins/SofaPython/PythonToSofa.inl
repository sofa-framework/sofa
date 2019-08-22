#ifndef PYTHON_TO_SOFA_INL
#define PYTHON_TO_SOFA_INL

#include <SofaPython/PythonMacros.h>
#include <type_traits>

namespace sofa {
namespace py {

/// casting PyObject* to Sofa types
/// contains only inlined functions and must be included at the right place (!)
template<class T, class = void>
struct wrap_traits {
    using wrapped_type = PyPtr<T>;
};


namespace detail {
template<class T, class Base>
using requires_derived = typename std::enable_if< std::is_base_of<Base, T>::value>::type;
}


// all base children are wrapped in a base
template<class T>
struct wrap_traits<T, detail::requires_derived<T, sofa::core::objectmodel::Base> > {
    using wrapped_type = PySPtr<sofa::core::objectmodel::Base>;
};


// all data childen are wrapped in a basedata
template<class T>
struct wrap_traits<T, detail::requires_derived<T, sofa::core::objectmodel::BaseData> > {
    using wrapped_type = PyPtr<sofa::core::objectmodel::BaseData>;
};

// all link childen are wrapped in a baselink
template<class T>
struct wrap_traits<T, detail::requires_derived<T, sofa::core::objectmodel::BaseLink> > {
    using wrapped_type = PyPtr<sofa::core::objectmodel::BaseLink>;
};






namespace detail {


template<class W>
static inline W* unwrap(PyPtr<W>* obj) {
    return obj->object;
}


template<class W>
static inline W* unwrap(PySPtr<W>* obj) {
    return obj->object.get();
}



/// wrap a T* in a pyobject (via PySPtr, based on traits)
template<class W, class T>
static inline void wrap(PySPtr<W>* py_obj, T* obj) {
    py_obj->object = obj;
}


template<class W, class T>
static inline void wrap(PyPtr<W>* py_obj, T* obj, bool deletable) {
    py_obj->object = obj;
    py_obj->deletable = deletable;
}



/// unwrap a T* wrapped in a pyobject (via PyPtr, based on traits)
template<class T>
static inline typename std::enable_if< !wrap_traits<T>::use_sptr, PyObject*>::type
wrap(T* obj, PyTypeObject *pto, bool deletable) {
    using wrapped_type = typename wrap_traits<T>::wrapped_type;

    PyPtr<wrapped_type>* py_obj = (PyPtr<wrapped_type>*) PyType_GenericAlloc(pto, 0);
    py_obj->object = obj;
    py_obj->deletable = deletable;
    return (PyObject*)py_obj;
}

}


// unwrap a python pointer for an argument
template<class T>
static inline T* unwrap(PyObject* obj) {
    using wrapped_type = typename wrap_traits<T>::wrapped_type;
    wrapped_type* wrapped = reinterpret_cast<wrapped_type*>(obj);
    return dynamic_cast<T*>(detail::unwrap(wrapped));
}


template<class T>
static inline T* unwrap_self(PyObject* obj) {
    using wrapped_type = typename wrap_traits<T>::wrapped_type;
    wrapped_type* wrapped = reinterpret_cast<wrapped_type*>(obj);
    return static_cast<T*>(detail::unwrap(wrapped));
}


// wrap a python object into a python object of type pto. you may need to pass
// extra boolean 'deletable' to specify python ownership (true = python may
// delete) depending on the wrapping type
template<class T, class ... Args>
static inline PyObject* wrap(T* obj, PyTypeObject *pto, Args&& ... args) {
    using wrapped_type = typename wrap_traits<T>::wrapped_type;

    // alloc object
    wrapped_type* py_obj = reinterpret_cast<wrapped_type*>( PyType_GenericAlloc(pto, 0) );

    // setup wrapper
    detail::wrap(py_obj, obj, std::forward<Args>(args)...);
    
    return reinterpret_cast<PyObject*>(py_obj);
};






}
}

#endif
