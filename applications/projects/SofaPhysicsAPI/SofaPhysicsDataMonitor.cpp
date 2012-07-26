#include "SofaPhysicsAPI.h"
#include "SofaPhysicsDataMonitor_impl.h"

SofaPhysicsDataMonitor::SofaPhysicsDataMonitor()
    : impl(new Impl)
{
}

SofaPhysicsDataMonitor::~SofaPhysicsDataMonitor()
{
    delete impl;
}

const char* SofaPhysicsDataMonitor::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsDataMonitor::getID() ///< unique ID of this object
{
    return impl->getID();
}

const char* SofaPhysicsDataMonitor::getValue() ///< Get the value of the associated variable
{
    return impl->getValue();
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsDataMonitor::Impl::Impl()
{
}

SofaPhysicsDataMonitor::Impl::~Impl()
{
}

const char* SofaPhysicsDataMonitor::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsDataMonitor::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

const char* SofaPhysicsDataMonitor::Impl::getValue() ///< Get the value of the associated variable
{
    if (!sObj) return 0;
    return sObj->getValue();
}
