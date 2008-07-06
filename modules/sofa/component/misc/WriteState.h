#ifndef SOFA_COMPONENT_MISC_WRITESTATE_H
#define SOFA_COMPONENT_MISC_WRITESTATE_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/common/Visitor.h>

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

///Create WriteState component in the graph each time needed
class WriteStateCreator: public Visitor
{
public:
    WriteStateCreator() : sceneName(""), counterWriteState(0), createInMapping(false) {}
    WriteStateCreator(std::string &n, int c=0) { sceneName=n; counterWriteState=c; }
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n; }
    void setCounter(int c) { counterWriteState = c; }
    void setCreateInMapping(bool b) { createInMapping=b; }
protected:
    void addWriteState(sofa::core::componentmodel::behavior::BaseMechanicalState*ms, simulation::Node* gnode);

    std::string sceneName;
    int counterWriteState; //avoid to have two same files if two mechanical objects has the same name
    bool createInMapping;
};

class WriteStateActivator: public simulation::Visitor
{
public:
    WriteStateActivator( bool active) : state(active) {}
    virtual Result processNodeTopDown( simulation::Node*  );

    bool getState() const { return state; }
    void setState(bool active) { state=active; }
protected:
    void changeStateWriter(sofa::component::misc::WriteState *ws);

    bool state;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
