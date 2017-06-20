#ifndef PYTHON_TO_SOFA_INL
#define PYTHON_TO_SOFA_INL

#include <type_traits>

/// casting PyObject* to Sofa types
/// contains only inlined functions and must be included at the right place (!)
template<class T, class = void>
struct unwrap_traits {
    using wrapped_type = T;
    static const bool use_sptr = false;
};


template<class T, class Base>
using requires_derived = typename std::enable_if< std::is_base_of<Base, T>::value>::type;

template<class T>
struct unwrap_traits<T, requires_derived<T, sofa::core::objectmodel::Base> > {
    using wrapped_type = sofa::core::objectmodel::Base;
    static const bool use_sptr = true;
};


/// unwrap a T* wrapped in a pyobject (PySPtr)
template<class T>
static inline typename std::enable_if< unwrap_traits<T>::use_sptr, T*>::type
unwrap_aux(PyObject* obj, T*) {
    return ((PySPtr<T>*)obj)->object.get();
}


/// unwrap a T* wrapped in a pyobject (PyPtr)
template<class T>
static inline T* unwrap_aux(PyObject* obj, ...) {
    return ((PyPtr<T>*)obj)->object;
}


template<class T>
static inline T* unwrap(PyObject* obj) {
    return unwrap_aux<T>(obj, 0);
}


/// get a self object from a wrapped base object
template<class T, class Base = T>
static inline T* get_self(PyObject* obj) {
    Base* base = unwrap<Base>(obj);
    return static_cast<T*>(base);
}


/// get a function argument from a wrapped base object
template<class T, class Wrapped = typename unwrap_traits<T>::wrapped_type>
static inline T* get_arg(PyObject* obj) {
    Wrapped* wrapped = unwrap<Wrapped>(obj);
    return dynamic_cast<T*>(wrapped);
}


/// getting a Base::SPtr from a PyObject*
static inline sofa::core::objectmodel::Base* get_base(PyObject* obj) {
    return unwrap<sofa::core::objectmodel::Base>( obj );
}


/// getting a BaseObject* from a PyObject*
static inline sofa::core::objectmodel::BaseObject* get_baseobject(PyObject* obj) {
    return get_base( obj )->toBaseObject();
}


/// getting a BaseContext* from a PyObject*
static inline sofa::core::objectmodel::BaseContext* get_basecontext(PyObject* obj) {
    return get_base( obj )->toBaseContext();
}


/// getting a BaseNode* from a PyObject*
static inline sofa::core::objectmodel::BaseNode* get_basenode(PyObject* obj) {
    return get_base( obj )->toBaseNode();
}


/// getting a Node* from a PyObject*
static inline sofa::simulation::Node* get_node(PyObject* obj) {
    return down_cast<sofa::simulation::Node>(get_base( obj )->toBaseNode());
}


/// getting a BaseMapping* from a PyObject*
static inline sofa::core::BaseMapping* get_basemapping(PyObject* obj) {
    return get_base( obj )->toBaseMapping();
}


/// getting a BaseState* from a PyObject*
static inline sofa::core::BaseState* get_basestate(PyObject* obj) {
    return get_base( obj )->toBaseState();
}


/// getting a DataEngine* from a PyObject*
static inline sofa::core::DataEngine* get_dataengine(PyObject* obj) {
    return get_base( obj )->toDataEngine();
}


/// getting a BaseLoader* from a PyObject*
static inline sofa::core::loader::BaseLoader* get_baseloader(PyObject* obj) {
    return get_base( obj )->toBaseLoader();
}


/// getting a BaseMechanicalState* from a PyObject*
static inline sofa::core::behavior::BaseMechanicalState* get_basemechanicalstate(PyObject* obj) {
    return get_base( obj )->toBaseMechanicalState();
}


/// getting a OdeSolver* from a PyObject*
static inline sofa::core::behavior::OdeSolver* get_odesolver(PyObject* obj) {
    return get_base( obj )->toOdeSolver();
}


/// getting a Topology* from a PyObject*
static inline sofa::core::topology::Topology* get_topology(PyObject* obj) {
    return get_base( obj )->toTopology();
}


/// getting a BaseMeshTopology* from a PyObject*
static inline sofa::core::topology::BaseMeshTopology* get_basemeshtopology(PyObject* obj) {
    return get_base( obj )->toBaseMeshTopology();
}


/// getting a VisualModel* from a PyObject*
static inline sofa::core::visual::VisualModel* get_visualmodel(PyObject* obj) {
    return get_base( obj )->toVisualModel();
}


/// getting a BaseData* from a PyObject*
static inline sofa::core::objectmodel::BaseData* get_basedata(PyObject* obj) {
    return unwrap<sofa::core::objectmodel::BaseData>(obj);
}


/// getting a DataFileName* from a PyObject*
static inline sofa::core::objectmodel::DataFileName* get_datafilename(PyObject* obj) {
    return down_cast<sofa::core::objectmodel::DataFileName>( get_basedata( obj ) );
}


/// getting a DataFileNameVector* from a PyObject*
static inline sofa::core::objectmodel::DataFileNameVector* get_datafilenamevector(PyObject* obj) {
    return down_cast<sofa::core::objectmodel::DataFileNameVector>( get_basedata( obj ) );
}


/// getting a BaseLink* from a PyObject*
static inline sofa::core::objectmodel::BaseLink* get_baselink(PyObject* obj) {
    return unwrap<sofa::core::objectmodel::BaseLink>(obj);
}


/// getting a Vector3* from a PyObject*
static inline sofa::defaulttype::Vector3* get_vector3(PyObject* obj) {
    return unwrap<sofa::defaulttype::Vector3>(obj);
}

#endif
