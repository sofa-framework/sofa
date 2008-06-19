#ifndef SOFA_COMPONENT_MISC_WRITESTATE_H
#define SOFA_COMPONENT_MISC_WRITESTATE_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/defaulttype/DataTypeInfo.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

/** Write State vectors to file at a given set of time instants
 * A period can be etablished at the last time instant
 * The DoFs to print can be chosen using DOFsX and DOFsV
 * Stop to write the state if the kinematic energy reach a given threshold (stopAt)
 * The energy will be measured at each period determined by keperiod
*/
class WriteState: public core::objectmodel::BaseObject
{
public:
    Data < std::string > f_filename;
    Data < bool > f_writeX;
    Data < bool > f_writeV;
    Data < double > f_interval;
    Data < helper::vector<double> > f_time;
    Data < double > f_period;
    Data < helper::vector<unsigned int> > f_DOFsX;
    Data < helper::vector<unsigned int> > f_DOFsV;
    Data < double > f_stopAt;
    Data < double > f_keperiod;

protected:
    core::componentmodel::behavior::BaseMechanicalState* mmodel;
    std::ofstream* outfile;
    unsigned int nextTime;
    double lastTime;
    bool kineticEnergyThresholdReached;
    double timeToTestEnergyIncrease;
    double savedKineticEnergy;

public:
    WriteState();

    virtual ~WriteState();

    virtual void init();

    virtual void reset();

    virtual void handleEvent(sofa::core::objectmodel::Event* event);


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
