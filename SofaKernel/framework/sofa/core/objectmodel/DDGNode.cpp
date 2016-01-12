/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/DataEngine.h>
#ifdef SOFA_HAVE_BOOST_THREAD
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp> 
#endif
//#include <sofa/helper/BackTrace.h>

//#define SOFA_DDG_TRACE

namespace sofa
{

namespace core
{

namespace objectmodel
{

struct DDGNode::UpdateState
{
    sofa::helper::system::atomic<int> updateThreadID;
    sofa::helper::system::atomic<int> lastUpdateThreadID;
#ifdef SOFA_HAVE_BOOST_THREAD
    boost::mutex updateMutex;
#endif
    UpdateState()
    : updateThreadID(-1)
    , lastUpdateThreadID(-1)
    {}
};

/// Constructor
DDGNode::DDGNode()
    : inputs(initLink("inputs", "Links to inputs Data"))
    , outputs(initLink("outputs", "Links to outputs Data"))
    , updateStates(new UpdateState[SOFA_DATA_MAX_ASPECTS])
{
}

DDGNode::~DDGNode()
{
    for(DDGLinkIterator it=inputs.begin(); it!=inputs.end(); ++it)
        (*it)->doDelOutput(this);
    for(DDGLinkIterator it=outputs.begin(); it!=outputs.end(); ++it)
        (*it)->doDelInput(this);
    delete[] updateStates;
}

template<>
TClass<DDGNode,void>::TClass()
{
    DDGNode* ptr = NULL;
    namespaceName = Base::namespaceName(ptr);
    className = Base::className(ptr);
    templateName = Base::templateName(ptr);
    shortName = Base::shortName(ptr);
}

void DDGNode::setDirtyValue(const core::ExecParams* params)
{
    FlagType& dirtyValue = dirtyFlags[currentAspect(params)].dirtyValue;
    if (!dirtyValue.exchange_and_add(1))
    {

#ifdef SOFA_DDG_TRACE
        // TRACE LOG
        Base* owner = getOwner();
        if (owner)
            owner->sout << "Data " << getName() << " is now dirty." << owner->sendl;
#endif
        setDirtyOutputs(params);
    }
}

void DDGNode::setDirtyOutputs(const core::ExecParams* params)
{
    FlagType& dirtyOutputs = dirtyFlags[currentAspect(params)].dirtyOutputs;
    if (!dirtyOutputs.exchange_and_add(1))
    {
        for(DDGLinkIterator it=outputs.begin(params), itend=outputs.end(params); it != itend; ++it)
        {
            (*it)->setDirtyValue(params);
        }
    }
}

void DDGNode::doCleanDirty(const core::ExecParams* params, bool warnBadUse)
{
    //if (!params) params = core::ExecParams::defaultInstance();
    const int aspect = currentAspect(params);
    FlagType& dirtyValue = dirtyFlags[aspect].dirtyValue;
    if (!dirtyValue) // this node is not dirty, nothing to do
    {
        return;
    }

    UpdateState& state = updateStates[aspect];
    int updateThreadID = state.updateThreadID;
    if (updateThreadID != -1) // a thread is currently updating this node, dirty flags will be updated once it finishes.
    {
        return;
    }

    if (warnBadUse)
    {
        Base* owner = getOwner();
        if (owner)
        {
            owner->serr << "Data " << getName() << " deprecated cleanDirty() called. "
                "Instead of calling update() directly, requestUpdate() should now be called "
                "to manage dirty flags and provide thread safety." << getOwner()->sendl;
        }
        //sofa::helper::BackTrace::dump();
    }

#ifdef SOFA_HAVE_BOOST_THREAD
    // Here we know this thread does not own the lock (otherwise updateThreadID would not be -1 earlier), so we can take it
    boost::lock_guard<boost::mutex> guard(state.updateMutex);
#endif

    dirtyValue = 0;

#ifdef SOFA_DDG_TRACE
    Base* owner = getOwner();
    if (owner)
        owner->sout << "Data " << getName() << " has been updated." << owner->sendl;
#endif

    for(DDGLinkIterator it=inputs.begin(params), itend=inputs.end(params); it != itend; ++it)
        (*it)->dirtyFlags[aspect].dirtyOutputs = 0;
}

void DDGNode::cleanDirty(const core::ExecParams* params)
{
    doCleanDirty(params, true);
}

void DDGNode::forceCleanDirty(const core::ExecParams* params)
{
    doCleanDirty(params, false);
}

void DDGNode::requestUpdate(const core::ExecParams* params)
{
    setDirtyValue(params);
    requestUpdateIfDirty(params);
}

void DDGNode::requestUpdateIfDirty(const core::ExecParams* params)
{
    /*if (!params)*/ params = core::ExecParams::defaultInstance();
    const int aspect = currentAspect(params);
    FlagType& dirtyValue = dirtyFlags[aspect].dirtyValue;
    UpdateState& state = updateStates[aspect];
    const int currentThreadID = params->threadID();
    int updateThreadID = state.updateThreadID;
    if (updateThreadID == currentThreadID) // recursive call to update, ignoring
    {

        //if (getOwner())
        //    getOwner()->serr << "Data " << getName() << " recursive update() ignored." << getOwner()->sendl;
        return;
    }

    if (dirtyValue == 0)
    {
        if (getOwner())
            getOwner()->serr << "Data " << getName() << " requestUpdateIfDirty nothing to do." << getOwner()->sendl;
        return;
    }

    // Make sure all inputs are updated (before taking the lock)
    for(DDGLinkIterator it=inputs.begin(params), itend=inputs.end(params); it != itend; ++it)
        (*it)->updateIfDirty(params);

    if (dirtyValue == 0)
    {
        //if (getOwner())
        //    getOwner()->serr << "Data " << getName() << " requestUpdateIfDirty nothing to do after updating inputs." << getOwner()->sendl;
        return;
    }

#ifdef SOFA_HAVE_BOOST_THREAD
    // Here we know this thread does not own the lock (otherwise updateThreadID would be currentThreadID earlier), so we can take it
    boost::lock_guard<boost::mutex> guard(state.updateMutex);
#endif
    if (dirtyValue != 0)
    {
        // we need to call update

        state.updateThreadID = currentThreadID; // store the thread ID to detect recursive calls
        state.lastUpdateThreadID = currentThreadID;

        update();

        // dirtyValue is updated only once update() is complete, so that other threads know that they need to wait for it
        dirtyValue = 0;

#ifdef SOFA_DDG_TRACE
        Base* owner = getOwner();
        if (owner)
            owner->sout << "Data " << getName() << " has been updated." << owner->sendl;
#endif

        for(DDGLinkIterator it=inputs.begin(params), itend=inputs.end(params); it != itend; ++it)
            (*it)->dirtyFlags[aspect].dirtyOutputs = 0;
        
        state.updateThreadID = -1;
    }
    else // else nothing to do, as another thread already updated this while we were waiting to acquire the lock
    {
#ifdef SOFA_DDG_TRACE
        if (getOwner())
            getOwner()->serr << "Data " << getName() << " update() from multiple threads (" << state.lastUpdateThreadID << " and " << currentThreadID << ")" << getOwner()->sendl;
#endif
    }

}

void DDGNode::copyAspect(int destAspect, int srcAspect)
{
    dirtyFlags[destAspect] = dirtyFlags[srcAspect];
}

void DDGNode::addInput(DDGNode* n)
{
    doAddInput(n);
    n->doAddOutput(this);
    setDirtyValue();
}

void DDGNode::delInput(DDGNode* n)
{
    doDelInput(n);
    n->doDelOutput(this);
}

void DDGNode::addOutput(DDGNode* n)
{
    doAddOutput(n);
    n->doAddInput(this);
    n->setDirtyValue();
}

void DDGNode::delOutput(DDGNode* n)
{
    doDelOutput(n);
    n->doDelInput(this);
}

const DDGNode::DDGLinkContainer& DDGNode::getInputs()
{
    return inputs.getValue();
}

const DDGNode::DDGLinkContainer& DDGNode::getOutputs()
{
    return outputs.getValue();
}

sofa::core::objectmodel::Base* LinkTraitsPtrCasts<DDGNode>::getBase(sofa::core::objectmodel::DDGNode* n)
{
    if (!n) return NULL;
    return n->getOwner();
    //sofa::core::objectmodel::BaseData* d = dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
    //if (d) return d->getOwner();
    //return dynamic_cast<sofa::core::objectmodel::Base*>(n);
}

sofa::core::objectmodel::BaseData* LinkTraitsPtrCasts<DDGNode>::getData(sofa::core::objectmodel::DDGNode* n)
{
    if (!n) return NULL;
    return n->getData();
    //return dynamic_cast<sofa::core::objectmodel::BaseData*>(n);
}

bool DDGNode::findDataLinkDest(DDGNode*& ptr, const std::string& path, const BaseLink* link)
{
    std::string pathStr, dataStr(" "); // non-empty data to specify that a data name is optionnal for DDGNode (as it can be a DataEngine)
    if (link)
    {
        if (!link->parseString(path, &pathStr, &dataStr))
            return false;
    }
    else
    {
        if (!BaseLink::ParseString(path, &pathStr, &dataStr, this->getOwner()))
            return false;
    }
    bool self = (pathStr.empty() || pathStr == "[]");
    if (dataStr == "") // no Data -> we look for a DataEngine
    {
        if (self)
        {
            ptr = this;
            return true;
        }
        else
        {
            Base* owner = this->getOwner();
            DataEngine* obj = NULL;
            if (!owner)
                return false;
            if (!owner->findLinkDest(obj, path, link))
                return false;
            ptr = obj;
            return true;
        }
    }
    Base* owner = this->getOwner();
    if (!owner)
        return false;
    if (self)
    {
        ptr = owner->findData(dataStr);
        return (ptr != NULL);
    }
    else
    {
        Base* obj = NULL;
        if (!owner->findLinkDest(obj, BaseLink::CreateString(pathStr), link))
            return false;
        if (!obj)
            return false;
        ptr = obj->findData(dataStr);
        return (ptr != NULL);
    }
    return false;
}

void DDGNode::addLink(BaseLink* /*l*/)
{
    // the inputs and outputs links in DDGNode is manually added
    // once the link vectors are constructed in Base or BaseData
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
