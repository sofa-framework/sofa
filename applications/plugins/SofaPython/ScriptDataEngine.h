#ifndef SCRIPTDATAENGINE_H
#define SCRIPTDATAENGINE_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/simulation/Node.h>
#include "ScriptEvent.h"
#include "ScriptFunction.h"

namespace sofa
{

namespace component
{

namespace controller
{


class SOFA_SOFAPYTHON_API ScriptDataEngine : public core::DataEngine
{
public:
    SOFA_CLASS(ScriptDataEngine,core::DataEngine);

protected:
    ScriptDataEngine();
    virtual ~ScriptDataEngine();

public:


    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override ;

    virtual void init() override ;
    virtual void reinit() override ;


    virtual void update() override ;


protected:
    /// @name Script interface
    ///   Function that need to be implemented for each script language
    /// Typically, all "script_*" functions call the corresponding "*" function of the script, if it exists
    /// @{

    virtual void loadScript() = 0;      // load & bind functions

    virtual void script_update() = 0;
    virtual void script_init() = 0;
    virtual void script_parse() = 0;
    //virtual void script_onLoaded(sofa::simulation::Node* node) = 0;   // called once, immediately after the script is loaded

};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SCRIPTDATAENGINE_H
