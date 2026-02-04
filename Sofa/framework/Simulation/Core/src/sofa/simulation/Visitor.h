/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/simulation/config.h>
#include <sofa/simulation/fwd.h>
#include <sofa/core/fwd.h>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/core/objectmodel/TagSet.h>
#include <sofa/helper/system/thread/CTime.h>

#include <string>
#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/core/VecId.h>
#endif


namespace sofa::simulation
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
    };
    typedef helper::system::thread::ctime_t ctime_t;
#ifdef SOFA_DUMP_VISITOR_INFO
    typedef sofa::helper::system::thread::CTime CTime;
#endif

    explicit Visitor(const sofa::core::ExecParams* params);
    virtual ~Visitor();

    const sofa::core::ExecParams* execParams() const { return params; }

    enum Result { RESULT_CONTINUE, RESULT_PRUNE };

    /// Callback method called when descending to a new node. Recursion will stop if this method returns RESULT_PRUNE
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

protected:
    void debug_write_state_before( sofa::core::objectmodel::BaseObject* obj ) ;
    void debug_write_state_after( sofa::core::objectmodel::BaseObject* obj ) ;

    /// Function to be called when a visitor executes a main task
    /// It surrounds the task function with debug information
    template< class VisitorType, class VContext, class ObjectType>
    void runVisitorTask(VisitorType *visitor,
                               VContext *ctx,
                               void (VisitorType::*task)(VContext *, ObjectType *),
                               ObjectType *ptr,
                               const std::string &typeInfo = std::string("type"));

    /// Function to be called when a visitor executes a main task
    /// It surrounds the task function with debug information
    template< class VisitorType, class VContext, class ObjectType>
    Result runVisitorTask(VisitorType *visitor,
                                 VContext *ctx,
                                 Result (VisitorType::*task)(VContext *, ObjectType *),
                                 ObjectType *ptr,
                                 const std::string &typeInfo = std::string("type"));

    template < class Visit, class VContext, class Container, typename PointedType = typename Container::pointed_type >
    void for_each(Visit* visitor,
                         VContext* ctx,
                         const Container& list,
                         void (Visit::*task)(VContext*, PointedType*),
                         const std::string &typeInfo = std::string("type"));

    template < class Visit, class VContext, class Container, typename PointedType = typename Container::pointed_type>
    Visitor::Result for_each(Visit* visitor,
                                    VContext* ctx,
                                    const Container& list,
                                    Visitor::Result (Visit::*task)(VContext*, PointedType*),
                                    const std::string &typeInfo = std::string("type"));

public:

    //method to compare the tags of the object with the ones of the visitor
    // return true if the object has all the tags of the visitor
    // or if no tag is set to the visitor
    bool testTags(sofa::core::objectmodel::BaseObject* obj);

    /// Alias for context->executeVisitor(this)
    virtual void execute(sofa::core::objectmodel::BaseContext* node, bool precomputedOrder=false);

    /// Optional helper method to call before handling an object if not using the for_each method.
    /// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
    virtual ctime_t begin(simulation::Node *node, sofa::core::objectmodel::BaseObject *obj,
                          const std::string &typeInfo = std::string("type"));

    /// Optional helper method to call after handling an object if not using the for_each method.
    /// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
    virtual void end(simulation::Node* node, sofa::core::objectmodel::BaseObject* obj, ctime_t t0);

    /// Optional helper method to call before handling an object if not using the for_each method.
    /// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
    virtual ctime_t begin(simulation::Visitor::VisitorContext *node, sofa::core::objectmodel::BaseObject *obj,
                  const std::string &typeInfo = std::string("type"));

    /// Optional helper method to call after handling an object if not using the for_each method.
    /// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
    virtual void end(simulation::Visitor::VisitorContext* node, sofa::core::objectmodel::BaseObject* obj, ctime_t t0);

    /// Specify whether this visitor can be parallelized.
    virtual bool isThreadSafe() const { return false; }

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
    const sofa::core::ExecParams* params;


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

    static void printVector(sofa::core::behavior::BaseMechanicalState *mm, sofa::core::ConstVecId id);

    virtual void printInfo(const sofa::core::objectmodel::BaseContext* context, bool dirDown);

    void setNode(sofa::core::objectmodel::Base* c);

    static void EnableExportStateVector(bool activation) {outputStateVector=activation;}
    static void SetFirstIndexStateVector(unsigned int first) {firstIndexStateVector=first;}
    static void SetRangeStateVector(int range) {rangeStateVector=range;}

    static bool IsExportStateVectorEnabled() {return outputStateVector;}
    static unsigned int GetFirstIndexStateVector() { return firstIndexStateVector;}
    static int GetRangeStateVector() {return rangeStateVector;}
protected:

    static std::ostream *outputVisitor;  //Output stream to dump the info
    static bool printActivated;          //bool to know if the stream is opened or not
    static bool outputStateVector;       //bool to know if we trace the evolution of the state vectors
    static unsigned int firstIndexStateVector; //numero of the first index of the particules to trace
    static int rangeStateVector;         //number of particules to trace
    static ctime_t initDumpTime;
    static std::vector< ctime_t > initNodeTime;

    sofa::core::objectmodel::Base* enteringBase;
    bool infoPrinted;

private:
    static void dumpInfo( const std::string &info);
#endif
};

template< class VisitorType, class VContext, class ObjectType>
void Visitor::runVisitorTask(VisitorType *visitor,
                             VContext *ctx,
                             void (VisitorType::*task)(VContext *, ObjectType *),
                             ObjectType *ptr,
                             const std::string &typeInfo)
{
    if(this->testTags(ptr))
    {
#ifdef SOFA_VERBOSE_TRAVERSAL
        visitor->debug_write_state_before(ptr);
#endif
        auto t = this->begin(ctx, ptr, typeInfo);
        (visitor->*task)(ctx, ptr);
        this->end(ctx, ptr, t);
#ifdef SOFA_VERBOSE_TRAVERSAL
        visitor->debug_write_state_after(ptr);
#endif
    }
}

template< class VisitorType, class VContext, class ObjectType>
Visitor::Result Visitor::runVisitorTask(VisitorType *visitor,
                                        VContext *ctx,
                                        Result (VisitorType::*task)(VContext *, ObjectType *),
                                        ObjectType *ptr,
                                        const std::string &typeInfo)
{
    Result res = Result::RESULT_CONTINUE;
    if(this->testTags(ptr))
    {
#ifdef SOFA_VERBOSE_TRAVERSAL
        visitor->debug_write_state_before(ptr);
#endif
        auto t = this->begin(ctx, ptr, typeInfo);
        res = (visitor->*task)(ctx, ptr);
        this->end(ctx, ptr, t);
#ifdef SOFA_VERBOSE_TRAVERSAL
        visitor->debug_write_state_after(ptr);
#endif
    }
    return res;
}

template < class VisitorType, class VContext, class Container, typename PointedType >
void Visitor::for_each(VisitorType *visitor,
                       VContext *ctx,
                       const Container &list,
                       void (VisitorType::*task)(VContext *, PointedType *),
                       const std::string &typeInfo)
{
    for (const auto& element : list)
    {
        runVisitorTask(visitor, ctx, task, &*element, typeInfo);
    }
}

template < class VisitorType, class VContext, class Container, typename PointedType >
Visitor::Result Visitor::for_each(VisitorType *visitor,
                                  VContext *ctx,
                                  const Container &list,
                                  Visitor::Result (VisitorType::*task)(VContext *, PointedType *),
                                  const std::string &typeInfo)
{
    Visitor::Result res = Visitor::RESULT_CONTINUE;
    for (const auto& element : list)
    {
        res = runVisitorTask(visitor, ctx, task, &*element, typeInfo);
    }
    return res;
}

} // namespace sofa::simulation
