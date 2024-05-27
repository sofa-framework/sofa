/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/playback/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/objectmodel/DataFileName.h>

#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
#include <zlib.h>
#endif

#include <fstream>

namespace sofa::component::playback
{

/** Write State vectors to file at a given set of time instants
 * A period can be etablished at the last time instant
 * The DoFs to print can be chosen using DOFsX and DOFsV
 * Stop to write the state if the kinematic energy reach a given threshold (stopAt)
 * The energy will be measured at each period determined by keperiod
*/
class SOFA_COMPONENT_PLAYBACK_API WriteState: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(WriteState,core::objectmodel::BaseObject);

    sofa::core::objectmodel::DataFileName d_filename;
    Data < bool > d_writeX; ///< flag enabling output of X vector
    Data < bool > d_writeX0; ///< flag enabling output of X0 vector
    Data < bool > d_writeV; ///< flag enabling output of V vector
    Data < bool > d_writeF; ///< flag enabling output of F vector
    Data < type::vector<double> > d_time; ///< set time to write outputs
    Data < double > d_period; ///< period between outputs
    Data < type::vector<unsigned int> > d_DOFsX; ///< set the position DOFs to write
    Data < type::vector<unsigned int> > d_DOFsV; ///< set the velocity DOFs to write
    Data < double > d_stopAt; ///< stop the simulation when the given threshold is reached
    Data < double > d_keperiod; ///< set the period to measure the kinetic energy increase

protected:
    core::behavior::BaseMechanicalState* mmodel;
    std::ofstream* outfile;
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    gzFile gzfile;
#endif
    unsigned int nextIteration;
    double lastTime;
    bool kineticEnergyThresholdReached;
    double timeToTestEnergyIncrease;
    double savedKineticEnergy;
    bool firstExport;
    bool periodicExport;
    bool validInit;


    WriteState();

    ~WriteState() override;
public:
    void init() override;

    void reinit() override;

    void reset() override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMechanicalState() == nullptr)
        {
            arg->logError("No mechanical state found in the context node.");
            return false;
        }
        return BaseObject::canCreate(obj, context, arg);
    }

};

///Create WriteState component in the graph each time needed
class SOFA_COMPONENT_PLAYBACK_API WriteStateCreator: public simulation::Visitor
{
public:
    WriteStateCreator(const core::ExecParams* params);
    WriteStateCreator(const core::ExecParams* params, const std::string &_sceneName, bool _recordX, bool _recordV, bool _recordF, bool _createInMapping, int _counterState=0);
    Result processNodeTopDown( simulation::Node*  ) override;

    void setSceneName(std::string &n) { sceneName = n; }
    void setRecordX(bool b) {recordX=b;}
    void setRecordV(bool b) {recordV=b;}
    void setRecordF(bool b) {recordF=b;}
    void setCreateInMapping(bool b) { createInMapping=b; }
    void setCounter(int c) { counterWriteState = c; }
    const char* getClassName() const override { return "WriteStateCreator"; }

    void setExportTimes(const type::vector<double>& times) { m_times = times; }
    void setPeriod(double period) { m_period = period; }
protected:
    std::string sceneName;
    std::string extension;
    type::vector<double> m_times;

    bool recordX,recordV,recordF;
    bool createInMapping;

    int counterWriteState; //avoid to have two same files if two mechanical objects has the same name
    double m_period = 0.0;

    void addWriteState(sofa::core::behavior::BaseMechanicalState*ms, simulation::Node* gnode);

};

class SOFA_COMPONENT_PLAYBACK_API WriteStateActivator: public simulation::Visitor
{
public:
    WriteStateActivator( const core::ExecParams* params, bool active) : Visitor(params), state(active) {}
    Result processNodeTopDown( simulation::Node*  ) override;

    bool getState() const { return state; }
    void setState(bool active) { state=active; }
    const char* getClassName() const override { return "WriteStateActivator"; }
protected:
    void changeStateWriter(sofa::component::playback::WriteState *ws);

    bool state;
};

} // namespace sofa::component::playback
