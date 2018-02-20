/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MISC_WRITESTATE_H
#define SOFA_COMPONENT_MISC_WRITESTATE_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/Visitor.h>

#ifdef SOFA_HAVE_ZLIB
#include <zlib.h>
#endif

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
class SOFA_EXPORTER_API WriteState: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(WriteState,core::objectmodel::BaseObject);

    sofa::core::objectmodel::DataFileName f_filename;
    Data < bool > f_writeX; ///< flag enabling output of X vector
    Data < bool > f_writeX0; ///< flag enabling output of X0 vector
    Data < bool > f_writeV; ///< flag enabling output of V vector
    Data < bool > f_writeF; ///< flag enabling output of F vector
    Data < double > f_interval; ///< time duration between outputs
    Data < helper::vector<double> > f_time; ///< set time to write outputs
    Data < double > f_period; ///< period between outputs
    Data < helper::vector<unsigned int> > f_DOFsX; ///< set the position DOFs to write
    Data < helper::vector<unsigned int> > f_DOFsV; ///< set the velocity DOFs to write
    Data < double > f_stopAt; ///< stop the simulation when the given threshold is reached
    Data < double > f_keperiod; ///< set the period to measure the kinetic energy increase

protected:
    core::behavior::BaseMechanicalState* mmodel;
    std::ofstream* outfile;
#ifdef SOFA_HAVE_ZLIB
    gzFile gzfile;
#endif
    unsigned int nextTime;
    double lastTime;
    bool kineticEnergyThresholdReached;
    double timeToTestEnergyIncrease;
    double savedKineticEnergy;


    WriteState();

    virtual ~WriteState();
public:
    virtual void init() override;

    virtual void reinit() override;

    virtual void reset() override;

    virtual void handleEvent(sofa::core::objectmodel::Event* event) override;


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMechanicalState() == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

};

///Create WriteState component in the graph each time needed
class SOFA_EXPORTER_API WriteStateCreator: public simulation::Visitor
{
public:
    WriteStateCreator(const core::ExecParams* params);
    WriteStateCreator(const core::ExecParams* params, const std::string &n, bool _recordX, bool _recordV, bool _recordF, bool _createInMapping, int c=0);
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n; }
    void setRecordX(bool b) {recordX=b;}
    void setRecordV(bool b) {recordV=b;}
    void setRecordF(bool b) {recordF=b;}
    void setCreateInMapping(bool b) { createInMapping=b; }
    void setCounter(int c) { counterWriteState = c; }
    virtual const char* getClassName() const { return "WriteStateCreator"; }
protected:
    std::string sceneName;
    std::string extension;
    bool recordX,recordV,recordF;
    bool createInMapping;

    int counterWriteState; //avoid to have two same files if two mechanical objects has the same name

    void addWriteState(sofa::core::behavior::BaseMechanicalState*ms, simulation::Node* gnode);

};

class SOFA_EXPORTER_API WriteStateActivator: public simulation::Visitor
{
public:
    WriteStateActivator( const core::ExecParams* params, bool active) : Visitor(params), state(active) {}
    virtual Result processNodeTopDown( simulation::Node*  );

    bool getState() const { return state; }
    void setState(bool active) { state=active; }
    virtual const char* getClassName() const { return "WriteStateActivator"; }
protected:
    void changeStateWriter(sofa::component::misc::WriteState *ws);

    bool state;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
