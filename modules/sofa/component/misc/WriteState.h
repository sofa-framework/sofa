/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
    WriteStateCreator(): sceneName(""), recordX(true),recordV(true), createInMapping(false), counterWriteState(0) {};
    WriteStateCreator(std::string &n, bool _recordX, bool _recordV, bool _createInMapping, int c=0) :
        sceneName(n),recordX(_recordX),recordV(_recordV),createInMapping(_createInMapping),counterWriteState(c) { };
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n; }
    void setRecordX(bool b) {recordX=b;}
    void setRecordV(bool b) {recordV=b;}
    void setCreateInMapping(bool b) { createInMapping=b; }
    void setCounter(int c) { counterWriteState = c; }
protected:
    std::string sceneName;
    bool recordX,recordV;
    bool createInMapping;

    int counterWriteState; //avoid to have two same files if two mechanical objects has the same name

    void addWriteState(sofa::core::componentmodel::behavior::BaseMechanicalState*ms, simulation::Node* gnode);

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
