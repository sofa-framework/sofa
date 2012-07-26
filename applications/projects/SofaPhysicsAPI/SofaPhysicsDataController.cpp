#include "SofaPhysicsAPI.h"
#include "SofaPhysicsDataController_impl.h"

SofaPhysicsDataController::SofaPhysicsDataController()
    : impl(new Impl)
{
}

SofaPhysicsDataController::~SofaPhysicsDataController()
{
    delete impl;
}

const char* SofaPhysicsDataController::getName() ///< (non-unique) name of this object
{
    return impl->getName();
}

ID SofaPhysicsDataController::getID() ///< unique ID of this object
{
    return impl->getID();
}

void SofaPhysicsDataController::setValue(const char* v) ///< Set the value of the associated variable
{
    return impl->setValue(v);
}

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;


SofaPhysicsDataController::Impl::Impl()
{
}

SofaPhysicsDataController::Impl::~Impl()
{
}

const char* SofaPhysicsDataController::Impl::getName() ///< (non-unique) name of this object
{
    if (!sObj) return "";
    return sObj->getName().c_str();
}

ID SofaPhysicsDataController::Impl::getID() ///< unique ID of this object
{
    return sObj.get();
}

void SofaPhysicsDataController::Impl::setValue(const char* v) ///< Set the value of the associated variable
{
    if (sObj)
        sObj->setValue(v);
}
