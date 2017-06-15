

/// casting PyObject* to Sofa types



/// getting a T* (inherited from Base) from a PyObject*
/// @warning generic, default implementation using a dynamic_cast
template<class T>
inline T* get(PyObject* obj) {
    return dynamic_cast<T*>(((PySPtr<sofa::core::objectmodel::Base>*)obj)->object.get());
}

/// getting a Node* from a PyObject*
inline sofa::simulation::Node* get_node(PyObject* obj) {
    return down_cast<sofa::simulation::Node>(((PySPtr<sofa::core::objectmodel::Base>*)obj)->object->toBaseNode());
}

/// getting a BaseObject* from a PyObject*
inline sofa::core::objectmodel::BaseObject* get_baseobject(PyObject* obj) {
    return ((PySPtr<sofa::core::objectmodel::Base>*) obj)->object->toBaseObject();
}

/// getting a Base* from a PyObject*
inline sofa::core::objectmodel::Base* get_base(PyObject* obj) {
    return ((PySPtr<sofa::core::objectmodel::Base>*)obj)->object.get();
}

