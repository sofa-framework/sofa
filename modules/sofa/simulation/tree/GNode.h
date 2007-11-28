/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_SIMULATION_TREE_GNODE_H
#define SOFA_SIMULATION_TREE_GNODE_H

#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/Shader.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/MasterSolver.h>
#include <sofa/core/componentmodel/collision/Pipeline.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/tree/VisitorScheduler.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <stack>
#include <iostream>


using std::cout;
using std::endl;

namespace sofa
{

namespace simulation
{

namespace tree
{

class Visitor;
class MutationListener;

/** Define the structure of the scene. Contains (as pointer lists) Component objects and children GNode objects.
*/
class GNode : public core::objectmodel::Context, public core::objectmodel::BaseNode
{
public:
    GNode( const std::string& name="", GNode* parent=NULL  );

    virtual ~GNode();

    //virtual const char* getTypeName() const { return "GNODE"; }

    /// Add a child node
    virtual void addChild(GNode* node);

    /// Remove a child
    virtual void removeChild(GNode* node);

    /// Add a child node
    virtual void addChild(BaseNode* node);

    /// Remove a child node
    virtual void removeChild(BaseNode* node);

    virtual const BaseContext* getContext() const;
    virtual BaseContext* getContext();

    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual bool addObject(core::objectmodel::BaseObject* obj);

    /// Remove an object
    virtual bool removeObject(core::objectmodel::BaseObject* obj);

    /// Move a node from another node
    virtual void moveChild(GNode* obj);

    /// Move an object from another node
    virtual void moveObject(core::objectmodel::BaseObject* obj);

    /// Must be called after each graph modification. Do not call it directly, apply an InitVisitor instead.
    virtual void initialize();

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual core::objectmodel::BaseNode* getParent();

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual const core::objectmodel::BaseNode* getParent() const;

    /// @name Variables
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual core::objectmodel::BaseObject* getMechanicalState() const;

    /// Topology
    virtual core::objectmodel::BaseObject* getTopology() const;

    /// Dynamic Topology
    virtual core::objectmodel::BaseObject* getMainTopology() const;

    /// Shader
    virtual core::objectmodel::BaseObject* getShader() const;

    /// @}

    /// Update the whole context values, based on parent and local ContextObjects
    void updateContext();

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    void updateSimulationContext();

    /// Update the visual context values, based on parent and local ContextObjects
    void updateVisualContext();

    /// @name Visitors and graph traversal
    /// @{

    /// Execute a recursive action starting from this node
    virtual void executeVisitor(Visitor* action);

    /// Execute a recursive action starting from this node
    void execute(Visitor& action)
    {
        Visitor* p = &action;
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    void execute(Visitor* p)
    {
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    template<class Act>
    void execute()
    {
        Act action;
        Visitor* p = &action;
        executeVisitor(p);
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

    /// Return an object of this node deriving from a given class, or NULL if not found.
    /// Note that only the first object is returned.
    template<class Object>
    void getNodeObject(Object*& result)
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            Object* o = dynamic_cast<Object*>(*it);
            if (o != NULL)
            {
                result = o;
                return;
            }
        }
        result = NULL;
    }

    template<class Object>
    Object* getNodeObject()
    {
        Object* result;
        this->getNodeObject(result);
        return result;
    }

    /// Return an object of this node and sub-nodes deriving from a given class, or NULL if not found.
    /// Note that only the first object is returned.
    template<class Object>
    void getTreeObject(Object*& result)
    {
        this->getNodeObject(result);
        if (result != NULL) return;
        for (ChildIterator it = this->child.begin(); it != this->child.end(); ++it)
        {
            GNode* n = *it;
            n->getTreeObject(result);
            if (result != NULL) return;
        }
    }

    template<class Object>
    Object* getTreeObject()
    {
        Object* result;
        this->getTreeObject(result);
        return result;
    }

    /// Find a child node given its name
    GNode* getChild(const std::string& name);

    /// Get a descendant node given its name
    GNode* getTreeNode(const std::string& name);

    /// Propagate an event
    virtual void propagateEvent( core::objectmodel::Event* event );

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

    Sequence<core::objectmodel::BaseObject> object;
    typedef Sequence<core::objectmodel::BaseObject>::iterator ObjectIterator;

    Single<core::componentmodel::behavior::MasterSolver> masterSolver;
    Sequence<core::componentmodel::behavior::OdeSolver> solver;
    Single<core::componentmodel::behavior::BaseMechanicalState> mechanicalState;
    Single<core::componentmodel::behavior::BaseMechanicalMapping> mechanicalMapping;
    Single<core::componentmodel::behavior::BaseMass> mass;
    Single<core::componentmodel::topology::Topology> topology;
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
    Single<VisitorScheduler> actionScheduler;

    GNode* setDebug(bool);
    bool getDebug() const;

    Sequence<MutationListener> listener;

    void addListener(MutationListener* obj);

    void removeListener(MutationListener* obj);

    void setLogTime(bool);
    bool getLogTime() const { return logTime_; }

    typedef helper::system::thread::ctime_t ctime_t;

    struct NodeTimer
    {
        ctime_t tNode; ///< total time elapsed in the node
        ctime_t tTree; ///< total time elapsed in the branch (node and children)
        int nVisit;    ///< number of visit
    };

    struct ObjectTimer
    {
        ctime_t tObject; ///< total time elapsed in the object
        int nVisit;    ///< number of visit
    };

    /// Reset time logs
    void resetTime();

    /// Get total time log
    const NodeTimer& getTotalTime() const { return totalTime; }

    /// Get time log of all categories
    const std::map<std::string, NodeTimer>& getVisitorTime() const { return actionTime; }

    /// Get time log of a given category
    const NodeTimer& getVisitorTime(const std::string& s) { return actionTime[s]; }

    /// Get time log of a given category
    const NodeTimer& getVisitorTime(const char* s) { return actionTime[s]; }

    /// Get time log of all objects
    const std::map<std::string, std::map<core::objectmodel::BaseObject*, ObjectTimer> >& getObjectTime() const { return objectTime; }

    /// Get time log of all objects of a given category
    const std::map<core::objectmodel::BaseObject*, ObjectTimer>& getObjectTime(const std::string& s) { return objectTime[s]; }

    /// Get time log of all objects of a given category
    const std::map<core::objectmodel::BaseObject*, ObjectTimer>& getObjectTime(const char* s) { return objectTime[s]; }

    /// Get timer frequency
    ctime_t getTimeFreq() const;

    /// Log time spent on an action category, and the concerned object, plus remove the computed time from the parent caller object
    void addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent);

    /// Log time spent on an action category and the concerned object
    void addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj);

    /// Measure start time
    ctime_t startTime() const;

    /// Log time spent given a start time, an action category, and the concerned object
    ctime_t endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj);

    /// Log time spent given a start time, an action category, and the concerned object, plus remove the computed time from the parent caller object
    ctime_t endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent);

    /// Return the full path name of this node
    std::string getPathName() const;

protected:
    bool debug_;
    bool logTime_;

    /// @name Performance Timing Log
    /// @{

    std::stack<Visitor*> actionStack;
    NodeTimer totalTime;
    std::map<std::string, NodeTimer> actionTime;
    std::map<std::string, std::map<core::objectmodel::BaseObject*, ObjectTimer> > objectTime;

    /// @}

    void doAddChild(GNode* node);
    void doRemoveChild(GNode* node);
    void doAddObject(core::objectmodel::BaseObject* obj);
    void doRemoveObject(core::objectmodel::BaseObject* obj);

    void notifyAddChild(GNode* node);
    void notifyRemoveChild(GNode* node);
    void notifyAddObject(core::objectmodel::BaseObject* obj);
    void notifyRemoveObject(core::objectmodel::BaseObject* obj);
    void notifyMoveChild(GNode* node, GNode* prev);
    void notifyMoveObject(core::objectmodel::BaseObject* obj, GNode* prev);

    /// Execute a recursive action starting from this node.
    /// This method bypass the actionScheduler of this node if any.
    void doExecuteVisitor(Visitor* action);

    // VisitorScheduler can use doExecuteVisitor() method
    friend class VisitorScheduler;

};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
