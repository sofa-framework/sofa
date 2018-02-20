/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_VISITOR_H
#define SOFA_SIMULATION_VISITOR_H

#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/LocalStorage.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/ExecParams.h>

#include <sofa/helper/set.h>
#include <iostream>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/helper/system/thread/CTime.h>
#endif

namespace sofa
{

namespace simulation
{

class LocalStorage;

/// Base class for visitors propagated recursively through the scenegraph
class SOFA_SIMULATION_CORE_API Visitor
{
public:

    class VisitorContext
    {
    public:
        simulation::Node* root; ///< root node from which the visitor was executed
        simulation::Node* node; ///< current node
        SReal* nodeData;       ///< SReal value associated with this subtree. Set to NULL if node-specific data is not in use
    };
    typedef helper::system::thread::ctime_t ctime_t;
#ifdef SOFA_DUMP_VISITOR_INFO
    typedef sofa::helper::system::thread::CTime CTime;
#endif

    Visitor(const core::ExecParams* params);
    virtual ~Visitor();

    const core::ExecParams* execParams() const { return params; }

    enum Result { RESULT_CONTINUE, RESULT_PRUNE };

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(simulation::Node* /*node*/) { return RESULT_CONTINUE; }

    /// Callback method called after child node have been processed and before going back to the parent node.
    virtual void processNodeBottomUp(simulation::Node* /*node*/) {}

    /// Return true to reverse the order of traversal of child nodes
    virtual bool childOrderReversed(simulation::Node* /*node*/) { return false; }



    typedef enum{ NO_REPETITION=0, REPEAT_ALL, REPEAT_ONCE } TreeTraversalRepetition;
    /// @return @a treeTraversal returns true if and only if a tree traversal must be enforced (even for a DAG)
    /// @param repeat Tell if a node callback can be executed several times (at each traversal in diamond configurations)
    virtual bool treeTraversal(TreeTraversalRepetition& repeat) { repeat=NO_REPETITION; return false; }

    /// Return a category name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "default"; }

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    virtual const char* getClassName() const { return "Visitor"; }

    /// Return eventual information on the behavior of the visitor
    /// Only used for debugging / profiling purposes
    virtual std::string getInfos() const { return ""; }

#ifdef SOFA_VERBOSE_TRAVERSAL
    void debug_write_state_before( core::objectmodel::BaseObject* obj ) ;
    void debug_write_state_after( core::objectmodel::BaseObject* obj ) ;
#else
    inline void debug_write_state_before( core::objectmodel::BaseObject*  ) {}
    inline void debug_write_state_after( core::objectmodel::BaseObject*  ) {}
#endif

    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Visit, class VContext, class Container, class Object >
    void for_each(Visit* visitor, VContext* ctx, const Container& list, void (Visit::*fn)(VContext*, Object*))
    {
        for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
        {
            typename Container::pointed_type* ptr = &*(*it);
            if(testTags(ptr))
            {
                debug_write_state_before(ptr);
                ctime_t t=begin(ctx, ptr);
                (visitor->*fn)(ctx, ptr);
                end(ctx, ptr, t);
                debug_write_state_after(ptr);
            }
        }
    }

    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Visit, class VContext, class Container, class Object >
    Visitor::Result for_each_r(Visit* visitor, VContext* ctx, const Container& list, Visitor::Result (Visit::*fn)(VContext*, Object*))
    {
        Visitor::Result res = Visitor::RESULT_CONTINUE;
        for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
        {
            typename Container::pointed_type* ptr = &*(*it);
            if(testTags(ptr))
            {
                debug_write_state_before(ptr);
                ctime_t t=begin(ctx, ptr);
                res = (visitor->*fn)(ctx, ptr);
                end(ctx, ptr, t);
                debug_write_state_after(ptr);
            }
        }
        return res;

    }


    //method to compare the tags of the objet with the ones of the visitor
    // return true if the object has all the tags of the visitor
    // or if no tag is set to the visitor
    bool testTags(core::objectmodel::BaseObject* obj)
    {
        if(subsetsToManage.empty())
            return true;
        if (obj->getTags().includes(subsetsToManage)) // all tags in subsetsToManage must be included in the list of tags of the object
            return true;
        return false;
    }


    //template < class Visit, class Container, class Object >
    //void for_each(Visit* visitor, const Container& list, void (Visit::*fn)(Object))
    //{
    //	for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
    //	{
    //		(visitor->*fn)(*it);
    //	}
    //}


    /// Alias for context->executeVisitor(this)
    virtual void execute(core::objectmodel::BaseContext* node, bool precomputedOrder=false);

    virtual ctime_t begin(simulation::Node* node, core::objectmodel::BaseObject* obj
            , const std::string &typeInfo=std::string("type")
                         );
    virtual void end(simulation::Node* node, core::objectmodel::BaseObject* obj, ctime_t t0);
    ctime_t begin(simulation::Visitor::VisitorContext* node, core::objectmodel::BaseObject* obj
            , const std::string &typeInfo=std::string("type")
                 );
    void end(simulation::Visitor::VisitorContext* node, core::objectmodel::BaseObject* obj, ctime_t t0);



    /// Specify whether this visitor can be parallelized.
    virtual bool isThreadSafe() const { return false; }

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    /// This version is offered a LocalStorage to store temporary data
    virtual Result processNodeTopDown(simulation::Node* node, LocalStorage*) { return processNodeTopDown(node); }

    /// Callback method called after child node have been processed and before going back to the parent node.
    /// This version is offered a LocalStorage to store temporary data
    virtual void processNodeBottomUp(simulation::Node* node, LocalStorage*) { processNodeBottomUp(node); }

public:
    typedef sofa::core::objectmodel::Tag Tag;
    typedef sofa::core::objectmodel::TagSet TagSet;
    /// list of the subsets
    TagSet subsetsToManage;

    Visitor& setTags(const TagSet& t) { subsetsToManage = t; return *this; }
    Visitor& addTag(Tag t) { subsetsToManage.insert(t); return *this; }
    Visitor& removeTag(Tag t) { subsetsToManage.erase(t); return *this; }

	/// Can the visitor access sleeping nodes?
	bool canAccessSleepingNode;

protected:
    const core::ExecParams* params;


#ifdef SOFA_DUMP_VISITOR_INFO
public:
    static SReal getTimeSpent(ctime_t initTime, ctime_t endTime);
    static void startDumpVisitor(std::ostream *s, SReal time);
    static void stopDumpVisitor();
    static bool isPrintActivated() { return printActivated; }
    typedef std::vector< std::pair< std::string,std::string > > TRACE_ARGUMENT;
    static void printComment(const std::string &s);
    static void printNode(const std::string &type, const std::string &name=std::string(), const TRACE_ARGUMENT &arguments=TRACE_ARGUMENT() );
    static void printCloseNode(const std::string &type);
    // const char* versions allow to call the print methods without dynamically allocating a std::string in case print is disabled
    static void printComment(const char* s);
    static void printNode(const char* type, const std::string &name, const TRACE_ARGUMENT &arguments);
    static void printNode(const char* type, const std::string &name);
    static void printNode(const char* type);
    static void printCloseNode(const char* type);

    static void printVector(core::behavior::BaseMechanicalState *mm, core::ConstVecId id);

    virtual void printInfo(const core::objectmodel::BaseContext* context, bool dirDown);

    void setNode(core::objectmodel::Base* c);

    static void EnableExportStateVector(bool activation) {outputStateVector=activation;}
    static void SetFirstIndexStateVector(unsigned int first) {firstIndexStateVector=first;}
    static void SetRangeStateVector(int range) {rangeStateVector=range;}

    static bool IsExportStateVectorEnabled() {return outputStateVector;}
    static unsigned int GetFirstIndexStateVector() { return firstIndexStateVector;}
    static int GetRangeStateVector() {return rangeStateVector;}
protected:

    static std::ostream *outputVisitor;  //Ouput stream to dump the info
    static bool printActivated;          //bool to know if the stream is opened or not
    static bool outputStateVector;       //bool to know if we trace the evolution of the state vectors
    static unsigned int firstIndexStateVector; //numero of the first index of the particules to trace
    static int rangeStateVector;         //number of particules to trace
    static ctime_t initDumpTime;
    static std::vector< ctime_t > initNodeTime;

    core::objectmodel::Base* enteringBase;
    bool infoPrinted;

private:
    static void dumpInfo( const std::string &info);
#endif
};
} // namespace simulation

} // namespace sofa

#endif
