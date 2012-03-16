#ifndef SCRIPTCONTROLLER_H
#define SCRIPTCONTROLLER_H

#include <sofa/component/controller/Controller.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace component
{

namespace controller
{

class ScriptController : public Controller
{
public:
    SOFA_CLASS(ScriptController,Controller);

protected:
    ScriptController();


public:
    /// @name control
    ///   Basic control (from BaseObject)
    /// @{

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg );

    /// Initialization method called at graph creation and modification, during top-down traversal.
    virtual void init();

    /// Initialization method called at graph creation and modification, during bottom-up traversal.
//    virtual void bwdInit();

    /// Update method called when variables used in precomputation are modified.
//    virtual void reinit();

    /// Save the initial state for later uses in reset()
    virtual void storeResetState();

    /// Reset to initial state
    virtual void reset();

    /// Called just before deleting this object
    /// Any object in the tree bellow this object that are to be removed will be removed only after this call,
    /// so any references this object holds should still be valid.
    virtual void cleanup();

    /// @}



    /// @name Controller notifications
    /// @{

    virtual void onBeginAnimationStep(const double dt);

    /// @}



protected:

    /// @name Script interface
    ///   Function that need to be implemented for each script language
    /// Typically, all "script_*" functions call the corresponding "*" function of the script, if it exists
    /// @{

    virtual void loadScript() = 0;      // load & bind functions

    virtual void script_onLoaded(sofa::simulation::tree::GNode* node) = 0;   // called once, immediately after the script is loaded
    virtual void script_createGraph(sofa::simulation::tree::GNode* node) = 0;       // called when the script must create its graph
    virtual void script_initGraph(sofa::simulation::tree::GNode* node) = 0;         // called when the script must init its graph, once all the graph has been create

    virtual void script_storeResetState() = 0;
    virtual void script_reset() = 0;

    virtual void script_cleanup() = 0;

    /// called once per frame
    virtual void script_onBeginAnimationStep(const double dt) = 0;

    /// @}


};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SCRIPTCONTROLLER_H
