

/// casting PyObject* to Sofa types
/// contains only inlined functions and must be included at the right place (!)




/// getting a T::SPtr from a PyObject*
/// @warning for T inherited from Base
template<class T>
inline typename T::SPtr get_sptr(PyObject* obj) {
    return ((PySPtr<T>*)obj)->object;
}



/// getting a T* from a PyObject*
/// @warning not to use for T inherited from Base
template<class T>
inline T* get(PyObject* obj) {
    return ((PyPtr<T>*)obj)->object;
}





/// getting a Base::SPtr from a PyObject*
inline sofa::core::objectmodel::Base::SPtr get_basesptr(PyObject* obj) {
    return get_sptr<sofa::core::objectmodel::Base>( obj );
}

/// getting a BaseObject* from a PyObject*
inline sofa::core::objectmodel::BaseObject* get_baseobject(PyObject* obj) {
    return get_basesptr( obj )->toBaseObject();
}

/// getting a Base* from a PyObject*
inline sofa::core::objectmodel::Base* get_base(PyObject* obj) {
    return get_basesptr( obj ).get();
}

/// getting a BaseContext* from a PyObject*
inline sofa::core::objectmodel::BaseContext* get_basecontext(PyObject* obj) {
    return get_basesptr( obj )->toBaseContext();
}

/// getting a BaseNode* from a PyObject*
inline sofa::core::objectmodel::BaseNode* get_basenode(PyObject* obj) {
    return get_basesptr( obj )->toBaseNode();
}



/// getting a Node* from a PyObject*
inline sofa::simulation::Node* get_node(PyObject* obj) {
    return down_cast<sofa::simulation::Node>(get_basesptr( obj )->toBaseNode());
}





/// getting a BaseMapping* from a PyObject*
inline sofa::core::BaseMapping* get_basemapping(PyObject* obj) {
    return get_basesptr( obj )->toBaseMapping();
}


/// getting a BaseState* from a PyObject*
inline sofa::core::BaseState* get_basestate(PyObject* obj) {
    return get_basesptr( obj )->toBaseState();
}

/// getting a DataEngine* from a PyObject*
inline sofa::core::DataEngine* get_dataengine(PyObject* obj) {
    return get_basesptr( obj )->toDataEngine();
}





/// getting a BaseLoader* from a PyObject*
inline sofa::core::loader::BaseLoader* get_baseloader(PyObject* obj) {
    return get_basesptr( obj )->toBaseLoader();
}





/// getting a BaseMechanicalState* from a PyObject*
inline sofa::core::behavior::BaseMechanicalState* get_basemechanicalstate(PyObject* obj) {
    return get_basesptr( obj )->toBaseMechanicalState();
}

/// getting a OdeSolver* from a PyObject*
inline sofa::core::behavior::OdeSolver* get_odesolver(PyObject* obj) {
    return get_basesptr( obj )->toOdeSolver();
}



/// getting a Topology* from a PyObject*
inline sofa::core::topology::Topology* get_topology(PyObject* obj) {
    return get_basesptr( obj )->toTopology();
}

/// getting a BaseMeshTopology* from a PyObject*
inline sofa::core::topology::BaseMeshTopology* get_basemeshtopology(PyObject* obj) {
    return get_basesptr( obj )->toBaseMeshTopology();
}


/// getting a VisualModel* from a PyObject*
inline sofa::core::visual::VisualModel* get_visualmodel(PyObject* obj) {
    return get_basesptr( obj )->toVisualModel();
}





/// getting a BaseData* from a PyObject*
inline sofa::core::objectmodel::BaseData* get_basedata(PyObject* obj) {
    return get<sofa::core::objectmodel::BaseData>(obj);
}

/// getting a DataFileName* from a PyObject*
inline sofa::core::objectmodel::DataFileName* get_datafilename(PyObject* obj) {
    return down_cast<sofa::core::objectmodel::DataFileName>( get_basedata( obj ) );
}

/// getting a DataFileNameVector* from a PyObject*
inline sofa::core::objectmodel::DataFileNameVector* get_datafilenamevector(PyObject* obj) {
    return down_cast<sofa::core::objectmodel::DataFileNameVector>( get_basedata( obj ) );
}


/// getting a BaseLink* from a PyObject*
inline sofa::core::objectmodel::BaseLink* get_baselink(PyObject* obj) {
    return get<sofa::core::objectmodel::BaseLink>(obj);
}



/// getting a Vector3* from a PyObject*
inline sofa::defaulttype::Vector3* get_vector3(PyObject* obj) {
    return get<sofa::defaulttype::Vector3>(obj);
}

