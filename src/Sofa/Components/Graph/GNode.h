#ifndef SOFA_COMPONENTS_GRAPH_GNODE_H
#define SOFA_COMPONENTS_GRAPH_GNODE_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include "Sofa/Abstract/BaseNode.h"
#include "Sofa/Abstract/BehaviorModel.h"
#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/ContextObject.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/Context.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Core/Mapping.h"
#include "Sofa/Core/MechanicalMapping.h"
#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/InteractionForceField.h"
#include "Sofa/Core/Mass.h"
#include "Sofa/Core/Constraint.h"
#include "Sofa/Core/Topology.h"
#include "Sofa/Core/OdeSolver.h"
#include "Action.h"
#include <iostream>
using std::cout;
using std::endl;

namespace Sofa
{
namespace Abstract
{
class BaseObject;
}

namespace Components
{

namespace Graph
{

using namespace Abstract;
using namespace Core;

class Action;

class GNode : public BaseNode
{
public:
    GNode( const std::string& name="", GNode* parent=NULL  );

    virtual ~GNode();

    /// Add a child node and return this
    virtual BaseNode* addChild(GNode* node);

    /// Remove a child
    virtual void removeChild(GNode* node);

    /// Add a child node and return this
    virtual BaseNode* addChild(BaseNode* node);

    /// Remove a child node
    virtual void removeChild(BaseNode* node);

    const Core::Context* getContext() const;
    Core::Context* getContext();

    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual GNode* addObject(BaseObject* obj);

    /// Remove an object
    virtual void removeObject(BaseObject* obj);

    /// Connect all objects together. Must be called after each graph modification.
    virtual void init();

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual BaseNode* getParent();

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual const BaseNode* getParent() const;

    /// Get the context of the parent, if the parent exists and is a Context
    Context* getParentContext();

    /// Update the context values, based on parent and local ContextObjects
    void updateContext();


    /// @name Actions and graph traversal
    /// @{

    /// Execute a recursive action starting from this node
    virtual void execute(Action* action);



    /// Execute a recursive action starting from this node
    template<class Act>
    void execute(Act action)
    {
        Action* p = &action;
        execute(p);
    }

    /// Execute a recursive action starting from this node
    template<class Act>
    void execute()
    {
        Act action;
        Action* p = &action;
        execute(p);
    }

    /// List all objects of this node deriving from a given class
    template<class Object, class Container>
    void getNodeObjects(Container* list)
    {
        //list->insert(list->end(),this->object.begin(),this->object.end());
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            Object* o = dynamic_cast<Object*>(*it);
            if (o!=NULL)
                list->push_back(o);
        }
    }

    /// List all objects of this node and sub-nodes deriving from a given class
    template<class Object, class Container>
    void getTreeObjects(Container* list)
    {
        this->getNodeObjects<Object, Container>(list);
        for (ChildIterator it = this->child.begin(); it != this->child.end(); ++it)
        {
            GNode* n = *it;
            n->getTreeObjects<Object, Container>(list);
        }
    }


    /// @}

    /// Sequence class to hold a list of objects. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end).
    template < class T >
    class Sequence
    {
    protected:
        std::vector< T* > elems;
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
    public:
        typedef T* value_type;
        typedef typename std::vector< T* >::const_iterator iterator;

        iterator begin() const
        {
            return elems.begin();
        }
        iterator end() const
        {
            return elems.end();
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
        friend class GNode;
    };

    /// Class to hold 0-or-1 object. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end), plus an automatic convertion to one pointer.
    template < class T >
    class Single
    {
    protected:
        T* elems[2];
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
        friend class GNode;
    };

    Single<GNode> parent;
    Sequence<GNode> child;
    typedef Sequence<GNode>::iterator ChildIterator;

    Sequence<BaseObject> object;
    typedef Sequence<BaseObject>::iterator ObjectIterator;

    Single<BasicMechanicalModel> mechanicalModel;
    Single<BasicMechanicalMapping> mechanicalMapping;
    Single<OdeSolver> solver;
    Single<Mass> mass;
    Single<Topology> topology;
    Sequence<ForceField> forceField;
    Sequence<InteractionForceField> interactionForceField;
    Sequence<Constraint> constraint;
    Sequence<ContextObject> contextObject;

    Sequence<BasicMapping> mapping;
    Sequence<BehaviorModel> behaviorModel;
    Sequence<VisualModel> visualModel;
    Sequence<CollisionModel> collisionModel;

    GNode* setDebug(bool);
    bool getDebug() const;

protected:
    Context context_;
    bool debug_;


};

} // namespace Graph

} // namespace Components

} // namespace Sofa

#endif


