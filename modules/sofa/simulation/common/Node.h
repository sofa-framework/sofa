/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
 *                                                                             *
 * This library is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation; either version 2.1 of the License, or (at     *
 * your option) any later version.                                             *
 *                                                                             *
 * This library is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
 * for more details.                                                           *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this library; if not, write to the Free Software Foundation,     *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
 *******************************************************************************
 *                               SOFA :: Modules                               *
 *                                                                             *
 * Authors: The SOFA Team and external contributors (see Authors.txt)          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
//
// C++ Interface: Node
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_COMMON_NODE_H
#define SOFA_SIMULATION_COMMON_NODE_H

#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/Context.h>
// moved from GNode (27/04/08)
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/Shader.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/MasterSolver.h>
#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/simulation/common/common.h>
#include <sofa/simulation/common/MutationListener.h>
#include <sofa/simulation/common/VisitorScheduler.h>
#include <sofa/simulation/common/xml/Element.h>

namespace sofa
{
namespace simulation
{
class Visitor;
}
}
using sofa::simulation::Visitor;
using sofa::simulation::VisitorScheduler;

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/visual/VisualParams.h>
#include <string>
#include <stack>

namespace sofa
{

namespace simulation
{

/**
   Implements the object (component) management of the core::Context.
   Contains objects in lists and provides accessors.
   The other nodes are not visible (unknown scene graph).

   @author The SOFA team </www.sofa-framework.org>
 */
class SOFA_SIMULATION_COMMON_API Node : public core::objectmodel::BaseNode, public sofa::core::objectmodel::Context
{

public:
    SOFA_CLASS2(Node, BaseNode, Context);

    typedef sofa::core::visual::DisplayFlags DisplayFlags;

    Node(const std::string& name="");

    virtual ~Node();

    /// Create, add, then return the new child of this Node
    virtual Node* createChild(const std::string& nodeName)=0;

    /// @name High-level interface
    /// @{
    /// Initialize the components
    void init(const core::ExecParams* params);
    /// Apply modifications to the components
    void reinit(const core::ExecParams* params);
    /// Do one step forward in time
    void animate(const core::ExecParams* params /* PARAMS FIRST */, double dt);
    /// Draw the objects in an OpenGl context
    void glDraw(core::visual::VisualParams* params);
    /// @}

    /// @name Visitor handling
    /// @{

    /// Execute a recursive action starting from this node.
    /// This method bypasses the actionScheduler of this node if any.
    virtual void doExecuteVisitor(Visitor* action)=0;

    /// Execute a recursive action starting from this node
    void executeVisitor( simulation::Visitor* action);

    /// Execute a recursive action starting from this node
    void execute(simulation::Visitor& action)
    {
        simulation::Visitor* p = &action;
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    void execute(simulation::Visitor* p)
    {
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    template<class Act, class Params>
    void execute(const Params* params)
    {
        Act action(params);
        simulation::Visitor* p = &action;
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    template<class Act>
    void execute(core::visual::VisualParams* vparams)
    {
        Act action(vparams);
        simulation::Visitor* p = &action;
        executeVisitor(p);
    }
    /// @}

    /// @name Component containers
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
        //friend class Node;
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
        void clear() { elems.clear(); }
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
        //friend class Node;
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

    Sequence<Node> child;
    typedef Sequence<Node>::iterator ChildIterator;

    Sequence<core::objectmodel::BaseObject> object;
    typedef Sequence<core::objectmodel::BaseObject>::iterator ObjectIterator;

    Single<core::behavior::MasterSolver> masterSolver;
    Sequence<core::behavior::OdeSolver> solver;
    Sequence<core::behavior::ConstraintSolver> constraintSolver;
    Sequence<core::behavior::LinearSolver> linearSolver;
    Single<core::BaseState> state;
    Single<core::behavior::BaseMechanicalState> mechanicalState;
    Single<core::BaseMapping> mechanicalMapping;
    Single<core::behavior::BaseMass> mass;
    Single<core::topology::Topology> topology;
    Single<core::topology::BaseMeshTopology> meshTopology;
    Single<core::visual::Shader> shader;

    //warning : basic topology are not yet used in the release version
    Sequence<core::topology::BaseTopology> basicTopology;

    Sequence<core::behavior::BaseForceField> forceField;
    Sequence<core::behavior::BaseInteractionForceField> interactionForceField;
    Sequence<core::behavior::BaseProjectiveConstraintSet> projectiveConstraintSet;
    Sequence<core::behavior::BaseConstraintSet> constraintSet;
    Sequence<core::objectmodel::ContextObject> contextObject;
    Sequence<core::objectmodel::ConfigurationSetting> configurationSetting;

    Sequence<core::BaseMapping> mapping;
    Sequence<core::BehaviorModel> behaviorModel;
    Sequence<core::visual::VisualModel> visualModel;
    Sequence<core::visual::VisualManager> visualManager;
    Sequence<core::CollisionModel> collisionModel;

    Single<core::collision::Pipeline> collisionPipeline;
    Sequence<core::objectmodel::BaseObject> unsorted;

    Single<Node>                nodeInVisualGraph;
    Sequence<Node>              childInVisualGraph;
    Sequence<core::objectmodel::BaseObject> componentInVisualGraph;

    Sequence<core::visual::VisualModel> visualModelInVisualGraph;
    Sequence<core::BaseMapping> visualMappingInVisualGraph;
    /// @}


    /// @name Set/get objects
    /// @{

    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual bool addObject(core::objectmodel::BaseObject* obj);

    /// Remove an object
    virtual bool removeObject(core::objectmodel::BaseObject* obj);

    /// Move an object from another node
    virtual void moveObject(core::objectmodel::BaseObject* obj);

    /// Find an object given its name
    core::objectmodel::BaseObject* getObject(const std::string& name) const;

#ifdef SOFA_SMP
    /// Get first partition
    Iterative::IterativePartition* getFirstPartition();
#endif

    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const=0;

    /// Generic object access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, SearchDirection dir = SearchUp) const
    {
        return getObject(class_info, sofa::core::objectmodel::TagSet(), dir);
    }

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const=0;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const =0;

    /// Generic list of objects access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const
    {
        getObjects(class_info, container, sofa::core::objectmodel::TagSet(), dir);
    }




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

    /// List all objects of this node and sub-nodes deriving from a given class
    template<class Object, class Container>
    void getTreeObjects(Container* list)
    {
        this->get<Object, Container>(list, SearchDown);
    }

    /// Return an object of this node and sub-nodes deriving from a given class, or NULL if not found.
    /// Note that only the first object is returned.
    template<class Object>
    void getTreeObject(Object*& result)
    {
        result = this->get<Object>(SearchDown);
    }

    template<class Object>
    Object* getTreeObject()
    {
        return this->get<Object>(SearchDown);
    }

    /// Topology
    virtual core::topology::Topology* getTopology() const;

    /// Mesh Topology (unified interface for both static and dynamic topologies)
    virtual core::topology::BaseMeshTopology* getMeshTopology() const;

    /// Degrees-of-Freedom
    virtual core::objectmodel::BaseObject* getState() const;

    /// Mechanical Degrees-of-Freedom
    virtual core::objectmodel::BaseObject* getMechanicalState() const;

    /// Shader
    virtual core::objectmodel::BaseObject* getShader() const;

    /// Remove odesolvers and mastercontroler
    virtual void removeControllers();

    /// @}

    /// @name Time management
    /// @{

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


    /// Find a child node given its name
    Node* getChild(const std::string& name) const;

    /// Get a descendant node given its name
    Node* getTreeNode(const std::string& name) const;

    /// Get children nodes
    virtual const Children getChildren() const;

    /// Get timer frequency
    ctime_t getTimeFreq() const;

    /// Measure start time
    ctime_t startTime() const;

    /// Log time spent on an action category and the concerned object
    virtual void addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj);

    /// Log time spent given a start time, an action category, and the concerned object
    virtual ctime_t endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj);

    /// Log time spent on an action category, and the concerned object, plus remove the computed time from the parent caller object
    virtual void addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent);

    /// Log time spent given a start time, an action category, and the concerned object, plus remove the computed time from the parent caller object
    virtual ctime_t endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent);
    /// @}

    Node* setDebug(bool);
    bool getDebug() const;
    // debug
    void printComponents();

    const BaseContext* getContext() const;
    BaseContext* getContext();

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext();

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext();

    /// Called during initialization to corectly propagate the visual context to the children
    virtual void initVisualContext() {}

    /// Propagate an event
    virtual void propagateEvent(const core::ExecParams* params /* PARAMS FIRST  = sofa::core::ExecParams::defaultInstance()*/, core::objectmodel::Event* event);

    /// Update the visual context values, based on parent and local ContextObjects
    virtual void updateVisualContext();

    Single<VisitorScheduler> actionScheduler;

    // VisitorScheduler can use doExecuteVisitor() method
    friend class VisitorScheduler;

    /// Must be called after each graph modification. Do not call it directly, apply an InitVisitor instead.
    virtual void initialize();

    /// Called after initialization to set the default value of the visual context.
    virtual void setDefaultVisualContextValue();

    template <class RealObject>
    static void create( RealObject*& obj, sofa::simulation::xml::Element<sofa::core::objectmodel::BaseNode>*& arg);
protected:
    bool debug_;
    bool logTime_;

    /// @name Performance Timing Log
    /// @{

    NodeTimer totalTime;
    std::map<std::string, NodeTimer> actionTime;
    std::map<std::string, std::map<core::objectmodel::BaseObject*, ObjectTimer> > objectTime;

    /// @}

    virtual void doAddObject(core::objectmodel::BaseObject* obj);
    virtual void doRemoveObject(core::objectmodel::BaseObject* obj);


    std::stack<Visitor*> actionStack;

    virtual void notifyAddChild(Node* node);
    virtual void notifyRemoveChild(Node* node);
    virtual void notifyMoveChild(Node* node, Node* prev);
    virtual void notifyAddObject(core::objectmodel::BaseObject* obj);
    virtual void notifyRemoveObject(core::objectmodel::BaseObject* obj);
    virtual void notifyMoveObject(core::objectmodel::BaseObject* obj, Node* prev);


    BaseContext* _context;

    Sequence<MutationListener> listener;


    // Added by FF to model component dependencies
public:

    virtual void addListener(MutationListener* obj);
    virtual void removeListener(MutationListener* obj);
    /// Pairs representing component dependencies. First must be initialized before second.
    Data < sofa::helper::vector < std::string > > depend;
    /// Sort the components according to the dependencies expressed in Data depend.
    void sortComponents();


};

}

}

#endif
