#ifndef SOFAPHYSICSDATAMONITOR_IMPL_H
#define SOFAPHYSICSDATAMONITOR_IMPL_H

#include "SofaPhysicsAPI.h"
#include <SofaValidation/DataMonitor.h>

class SofaPhysicsDataMonitor::Impl
{
public:

    Impl();
    ~Impl();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    const char* getValue();   ///< Get the value of the associated variable

    typedef sofa::component::misc::DataMonitor SofaDataMonitor;

protected:
    SofaDataMonitor::SPtr sObj;

public:
    SofaDataMonitor* getObject() { return sObj.get(); }
    void setObject(SofaDataMonitor* dm) { sObj = dm; }
};

#endif // SOFAPHYSICSDATAMONITOR_IMPL_H
