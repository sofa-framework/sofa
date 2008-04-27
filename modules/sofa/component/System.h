//
// C++ Interface: System
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef sofa_componentSystem_h
#define sofa_componentSystem_h

#include <sofa/core/objectmodel/Context.h>
// moved from GNode (27/04/08)
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/Shader.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/MasterSolver.h>
#include <sofa/core/componentmodel/collision/Pipeline.h>

namespace sofa
{

namespace component
{

/**
Implements the object (component) management of the core::Context.
Contains objects in lists and provides accessors.
The other systems are not visible (unknown scene graph).

	@author The SOFA team </www.sofa-framework.org>
*/
class System : public sofa::core::objectmodel::Context, public core::objectmodel::BaseNode
{
public:
    System();

    virtual ~System();

    /// @name Component (=Object)
    /// @{
    // methods moved from GNode (27/04/08)

    /// Sequence class to hold a list of objects. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end).
    template < class T >
    class Sequence
    {
    protected:
        std::vector< T* > elems;
    public:
        typedef T* value_type;
        typedef typename std::vector< T* >::const_iterator iterator;
        typedef typename std::vector< T* >::const_reverse_iterator reverse_iterator;

        iterator begin() const
        {
            return elems.begin();
        }
        iterator end() const
        {
            return elems.end();
        }
        reverse_iterator rbegin() const
        {
            return elems.rbegin();
        }
        reverse_iterator rend() const
        {
            return elems.rend();
        }
        unsigned int size() const
        {
            return elems.size();
        }
        bool empty() const
        {
            return elems.empty();
        }
        T* operator[](unsigned int i) const
        {
            return elems[i];
        }
        /// Swap two values in the list. Uses a const_cast to violate the read-only iterators.
        void swap( iterator a, iterator b )
        {
            T*& wa = const_cast<T*&>(*a);
            T*& wb = const_cast<T*&>(*b);
            T* tmp = *a;
            wa = *b;
            wb = tmp;
        }
        //friend class System;
        bool add
        (T* elem)
        {
            if (elem == NULL)
                return false;
            elems.push_back(elem);
            return true;
        }
        bool remove
        (T* elem)
        {
            if (elem == NULL)
                return false;
            typename std::vector< T* >::iterator it = elems.begin();
            while (it != elems.end() && (*it)!=elem)
                ++it;
            if (it != elems.end())
            {
                elems.erase(it);
                return true;
            }
            else
                return false;
        }
    };

    /// Class to hold 0-or-1 object. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end), plus an automatic convertion to one pointer.
    template < class T >
    class Single
    {
    protected:
        T* elems[2];
    public:
        typedef T* value_type;
        typedef T* const * iterator;

        Single()
        {
            elems[0] = NULL;
            elems[1] = NULL;
        }
        iterator begin() const
        {
            return elems;
        }
        iterator end() const
        {
            return (elems[0]==NULL)?elems:elems+1;
        }
        unsigned int size() const
        {
            return (elems[0]==NULL)?0:1;
        }
        bool empty() const
        {
            return (elems[0]==NULL);
        }
        T* operator[](unsigned int i) const
        {
            return elems[i];
        }
        operator T*() const
        {
            return elems[0];
        }
        T* operator->() const
        {
            return elems[0];
        }
        //friend class System;
        bool add
        (T* elem)
        {
            if (elem == NULL)
                return false;
            elems[0] = elem;
            return true;
        }
        bool remove
        (T* elem)
        {
            if (elem == NULL)
                return false;
            if (elems[0] == elem)
            {
                elems[0] = NULL;
                return true;
            }
            else
                return false;
        }
    };

    Sequence<core::objectmodel::BaseObject> object;
    typedef Sequence<core::objectmodel::BaseObject>::iterator ObjectIterator;

    Single<core::componentmodel::behavior::MasterSolver> masterSolver;
    Sequence<core::componentmodel::behavior::OdeSolver> solver;
    Single<core::componentmodel::behavior::BaseMechanicalState> mechanicalState;
    Single<core::componentmodel::behavior::BaseMechanicalMapping> mechanicalMapping;
    Single<core::componentmodel::behavior::BaseMass> mass;
    Single<core::componentmodel::topology::Topology> topology;
    Single<core::componentmodel::topology::BaseMeshTopology> meshTopology;
    Single<sofa::core::Shader> shader;

    //warning : basic topology are not yet used in the release version
    Sequence<core::componentmodel::topology::BaseTopology> basicTopology;

    Sequence<core::componentmodel::behavior::BaseForceField> forceField;
    Sequence<core::componentmodel::behavior::InteractionForceField> interactionForceField;
    Sequence<core::componentmodel::behavior::BaseConstraint> constraint;
    Sequence<core::objectmodel::ContextObject> contextObject;

    Sequence<core::BaseMapping> mapping;
    Sequence<core::BehaviorModel> behaviorModel;
    Sequence<core::VisualModel> visualModel;
    Sequence<core::CollisionModel> collisionModel;

    Single<core::componentmodel::collision::Pipeline> collisionPipeline;


    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual bool addObject(core::objectmodel::BaseObject* obj);

    /// Remove an object
    virtual bool removeObject(core::objectmodel::BaseObject* obj);

    /// Move an object from another node
    virtual void moveObject(core::objectmodel::BaseObject* obj);

    /// Find an object given its name
    core::objectmodel::BaseObject* getObject(const std::string& name) const;

    /// List all objects of this node deriving from a given class
    template<class Object, class Container>
    void getNodeObjects(Container* list)
    {
        this->get<Object, Container>(list, Local);
    }

    /// Return an object of this node deriving from a given class, or NULL if not found.
    /// Note that only the first object is returned.
    template<class Object>
    void getNodeObject(Object*& result)
    {
        result = this->get<Object>(Local);
    }

    template<class Object>
    Object* getNodeObject()
    {
        return this->get<Object>(Local);
    }

    /// Mechanical Degrees-of-Freedom
    virtual core::objectmodel::BaseObject* getMechanicalState() const;

    /// Topology
    virtual core::componentmodel::topology::Topology* getTopology() const;

    /// Dynamic Topology
    virtual core::componentmodel::topology::BaseTopology* getMainTopology() const;

    /// Mesh Topology (unified interface for both static and dynamic topologies)
    virtual core::componentmodel::topology::BaseMeshTopology* getMeshTopology() const;

    /// Shader
    virtual core::objectmodel::BaseObject* getShader() const;

    /// @}

protected:
    virtual void doAddObject(core::objectmodel::BaseObject* obj);
    virtual void doRemoveObject(core::objectmodel::BaseObject* obj);
};

}

}

#endif
