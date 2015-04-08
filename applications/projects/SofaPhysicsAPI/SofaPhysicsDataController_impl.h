#ifndef SOFAPHYSICSDATACONTROLLER_IMPL_H
#define SOFAPHYSICSDATACONTROLLER_IMPL_H

#include "SofaPhysicsAPI.h"
#include <SofaValidation/DataController.h>

class SofaPhysicsDataController::Impl
{
public:

    Impl();
    ~Impl();

    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object
    /// Set the value of the associated variable
    void setValue(const char* v);

    typedef sofa::component::misc::DataController SofaDataController;

protected:
    SofaDataController::SPtr sObj;

public:
    SofaDataController* getObject() { return sObj.get(); }
    void setObject(SofaDataController* dc) { sObj = dc; }
};

#endif // SOFAPHYSICSDATAMONITOR_IMPL_H
