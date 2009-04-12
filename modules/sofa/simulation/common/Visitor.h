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
#ifndef SOFA_SIMULATION_VISITOR_H
#define SOFA_SIMULATION_VISITOR_H

#include <sofa/simulation/common/common.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/LocalStorage.h>

#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

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
class SOFA_SIMULATION_COMMON_API Visitor
{
protected:
    bool prefetching;
public:
#ifdef SOFA_DUMP_VISITOR_INFO
    typedef sofa::helper::system::thread::CTime CTime;
#endif

    Visitor()
        : prefetching(false)
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        enteringBase=NULL; infoPrinted=false;
#endif
    }
    virtual ~Visitor() {}
    typedef simulation::Node::ctime_t ctime_t;

    enum Result { RESULT_CONTINUE, RESULT_PRUNE };

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(simulation::Node* /*node*/) { return RESULT_CONTINUE; }

    /// Callback method called after child node have been processed and before going back to the parent node.
    virtual void processNodeBottomUp(simulation::Node* /*node*/) {}

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
    template < class Visit, class Container, class Object >
    void for_each(Visit* visitor, simulation::Node* node, const Container& list, void (Visit::*fn)(simulation::Node*, Object*))
    {

        if (node->getLogTime())
        {
            const std::string category = getCategoryName();
            ctime_t t0 = node->startTime();
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                if(testTags(*it))
                {
                    debug_write_state_before(*it);
                    (visitor->*fn)(node, *it);
                    debug_write_state_after(*it);
                    t0 = node->endTime(t0, category, *it);
                }
            }
        }
        else
        {
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                if(testTags(*it))
                {
                    debug_write_state_before(*it);
                    (visitor->*fn)(node, *it);
                    debug_write_state_after(*it);
                }
            }
        }
    }

    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Visit, class Container, class Object >
    Visitor::Result for_each_r(Visit* visitor, simulation::Node* node, const Container& list, Visitor::Result (Visit::*fn)(simulation::Node*, Object*))
    {

        Visitor::Result res = Visitor::RESULT_CONTINUE;
        if (node->getLogTime())
        {
            const std::string category = getCategoryName();
            ctime_t t0 = node->startTime();

            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                if(testTags(*it))
                {
                    debug_write_state_before(*it);
                    res = (visitor->*fn)(node, *it);
                    debug_write_state_after(*it);
                    t0 = node->endTime(t0, category, *it);
                }
            }
        }
        else
        {
            //bool processObject;
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                if(testTags(*it))
                {
                    debug_write_state_before(*it);
                    res = (visitor->*fn)(node, *it);
                    debug_write_state_after(*it);
                }
            }
        }
        return res;

    }


    //method to compare the tags of the objet with the ones of the visitor
    // return true if the object has all the tags of the visitor
    // or if no tag is set to the visitor
    bool testTags(core::objectmodel::BaseObject* obj)
    {
        if (prefetching && !obj->canPrefetch()) return false;
        if(subsetsToManage.empty())
            return true;
        else
        {
            //for ( sofa::helper::set<unsigned int>::iterator it=subsetsToManage.begin() ; it!=subsetsToManage.end() ; it++)
            //	if(obj->hasTag(*it))
            //		return true;
            if (obj->getTags().includes(subsetsToManage)) // all tags in subsetsToManage must be included in the list of tags of the object
                return true;
        }
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
    virtual void execute(core::objectmodel::BaseContext* node, bool doPrefetch);
    virtual void execute(core::objectmodel::BaseContext* node) { execute(node, false); }
    ctime_t begin(simulation::Node* node, core::objectmodel::BaseObject*
#ifdef SOFA_DUMP_VISITOR_INFO
            obj
#endif
                 );
    void end(simulation::Node* node, core::objectmodel::BaseObject* obj, ctime_t t0);


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

#ifdef SOFA_DUMP_VISITOR_INFO
    //DEBUG Purposes
    static double getTimeSpent(ctime_t initTime, ctime_t endTime)
    {
        return 1000.0*(endTime-initTime)/((double)CTime::getTicksPerSec());
    }
protected:

    static std::ostream *outputVisitor;  //Ouput stream to dump the info
    static bool printActivated;          //bool to know if the stream is opened or not
    static unsigned int depthLevel;      //Level in the hierarchy
    static ctime_t initDumpTime;
    static std::vector< ctime_t > initNodeTime;

    core::objectmodel::Base* enteringBase;
    bool infoPrinted;

public:
    static void startDumpVisitor(std::ostream *s, double time)
    {
        depthLevel=0;
        initDumpTime = sofa::helper::system::thread::CTime::getRefTime();
        printActivated=true; outputVisitor=s;
        std::string initDump;
        std::ostringstream ff; ff << "<TraceVisitor time=\"" << time << "\">\n";
        dumpInfo(ff.str()); depthLevel++;
    };
    static void stopDumpVisitor()
    {
        std::ostringstream s;
        s << "<TotalTime value=\"" << getTimeSpent(initDumpTime, sofa::helper::system::thread::CTime::getRefTime()) << "\" />\n";
        s << "</TraceVisitor>\n";
        depthLevel--;  dumpInfo(s.str());
        printActivated=false;
        depthLevel=0;
    };

    typedef std::vector< std::pair< std::string,std::string > > TRACE_ARGUMENT;
    static void dumpInfo( const std::string &info) { if (printActivated) {(*outputVisitor) << info; outputVisitor->flush();}}
    static void printComment(const std::string &s) ;
    static void printNode(const std::string &type, const std::string &name=std::string(), const TRACE_ARGUMENT &arguments=TRACE_ARGUMENT() ) ;
    static void printCloseNode(const std::string &type) ;
    static unsigned int getLevel() {return depthLevel;};
    static void resetLevel() {depthLevel=0;};
    virtual void printInfo(const core::objectmodel::BaseContext* context, bool dirDown);
    void setNode(core::objectmodel::Base* c);
#endif
};
} // namespace simulation

} // namespace sofa

#endif
