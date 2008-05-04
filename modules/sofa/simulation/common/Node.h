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
#ifndef sofa_componentNode_h
#define sofa_componentNode_h

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
#include <sofa/core/objectmodel/Event.h>

#include <sofa/simulation/tree/VisitorScheduler.h>
using sofa::simulation::tree::VisitorScheduler;
namespace sofa
{
namespace simulation
{
namespace tree
{
class Visitor;
}
}
}
using sofa::simulation::tree::Visitor;

#include <sofa/helper/system/thread/CTime.h>
#include <string>
#include <stack>

namespace sofa
{

namespace simulation
{

/**
Implements the object (component) management of the core::Context.
Contains objects in lists and provides accessors.
The other systems are not visible (unknown scene graph).

	@author The SOFA team </www.sofa-framework.org>
*/
class Node : public sofa::core::objectmodel::Context
{

public:
    Node(const std::string& name="");

    virtual ~Node();

    /// @name High-level interface
    /// @{
    /// Initialize the components
    void init();
    /// Do one step forward in time
    void animate( double dt );
    /// Draw the objects in an OpenGl context
    void glDraw();
    /// @}

    /// @name Visitor handling
    /// @{

    /// Execute a recursive action starting from this node.
    /// This method bypasses the actionScheduler of this node if any.
    virtual void doExecuteVisitor(Visitor* action)=0;

    /// Execute a recursive action starting from this node
    void executeVisitor( simulation::tree::Visitor* action);

    /// Execute a recursive action starting from this node
    void execute(simulation::tree::Visitor& action)
    {
        simulation::tree::Visitor* p = &action;
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    void execute(simulation::tree::Visitor* p)
    {
        executeVisitor(p);
    }

    /// Execute a recursive action starting from this node
    template<class Act>
    void execute()
    {
        Act action;
        simulation::tree::Visitor* p = &action;
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

    /// Topology
    virtual core::componentmodel::topology::Topology* getTopology() const;

    /// Dynamic Topology
    virtual core::componentmodel::topology::BaseTopology* getMainTopology() const;

    /// Mesh Topology (unified interface for both static and dynamic topologies)
    virtual core::componentmodel::topology::BaseMeshTopology* getMeshTopology() const;

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

    /*
    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual core::objectmodel::BaseNode* getParent();

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual const core::objectmodel::BaseNode* getParent() const;

    /// Get a list of child node
    virtual sofa::helper::vector< core::objectmodel::BaseNode* >  getChildren();

    /// Get a list of child node
    virtual const sofa::helper::vector< core::objectmodel::BaseNode* >  getChildren() const;
    */

    const BaseContext* getContext() const;
    BaseContext* getContext();

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext();

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext();

    /// Called during initialization to corectly propagate the visual context to the children
    virtual void initVisualContext() {}

    /// Propagate an event
    virtual void propagateEvent( core::objectmodel::Event* event );

    /// Update the visual context values, based on parent and local ContextObjects
    virtual void updateVisualContext(int FILTER=0);

    Single<VisitorScheduler> actionScheduler;

    // VisitorScheduler can use doExecuteVisitor() method
    friend class VisitorScheduler;

    /// Must be called after each graph modification. Do not call it directly, apply an InitVisitor instead.
    virtual void initialize();

    /// Called after initialization to set the default value of the visual context.
    virtual void setDefaultVisualContextValue();

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
    virtual void notifyAddObject(core::objectmodel::BaseObject* ) {}
    virtual void notifyRemoveObject(core::objectmodel::BaseObject* ) {}
    virtual void notifyMoveObject(core::objectmodel::BaseObject* , Node* /*prev*/) {}

    BaseContext* _context;

};

}

}

#endif
