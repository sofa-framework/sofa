#ifndef SOFA_COMPONENT_MISC_READSTATE_H
#define SOFA_COMPONENT_MISC_READSTATE_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

/** Read State vectors from file at each timestep
*/
class ReadState: public core::objectmodel::BaseObject
{
public:

    Data < std::string > f_filename;
    Data < double > f_interval;
    Data < double > f_shift;

protected:
    core::componentmodel::behavior::BaseMechanicalState* mmodel;
    std::ifstream* infile;
    double nextTime;
    double lastTime;
public:
    ReadState();

    virtual ~ReadState();

    virtual void init();

    virtual void reset();

    void setTime(double time);

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

    void processReadState();
    void processReadState(double time);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::componentmodel::behavior::BaseMechanicalState*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }


};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
