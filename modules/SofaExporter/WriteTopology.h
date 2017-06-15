/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_WRITETOPOLOGY_H
#define SOFA_COMPONENT_MISC_WRITETOPOLOGY_H
#include "config.h"

#include <sofa/core/topology/BaseMeshTopology.h>
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

/** Write Topology containers informations into a file at a given set of time instants
 * A period can be etablished at the last time instant.
 * The informations to write can be choosen. by default there will be only commun containers.
 * An option is available to write shells containers.
 *
 * This part is not handle yet:
 * Stop to write infos if the kinematic energy reach a given threshold (stopAt)
 * The energy will be measured at each period determined by keperiod
*/
class SOFA_EXPORTER_API WriteTopology: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(WriteTopology,core::objectmodel::BaseObject);

    sofa::core::objectmodel::DataFileName f_filename;
    Data < bool > f_writeContainers;
    Data < bool > f_writeShellContainers;
    Data < double > f_interval;
    Data < helper::vector<double> > f_time;
    Data < double > f_period;
    //    Data < helper::vector<unsigned int> > f_DOFsX;
    //    Data < helper::vector<unsigned int> > f_DOFsV;
    //    Data < double > f_stopAt;
    //    Data < double > f_keperiod;

protected:
    core::topology::BaseMeshTopology* m_topology;
    std::ofstream* outfile;
#ifdef SOFA_HAVE_ZLIB
    gzFile gzfile;
#endif
    unsigned int nextTime;
    double lastTime;
    //    bool kineticEnergyThresholdReached;
    //    double timeToTestEnergyIncrease;
    //    double savedKineticEnergy;


    WriteTopology();

    virtual ~WriteTopology();
public:
    virtual void init();

    virtual void reset();

    virtual void handleEvent(sofa::core::objectmodel::Event* event);


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MeshTopology.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMeshTopology() == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

};


///Create WriteTopology component in the graph each time needed
class SOFA_EXPORTER_API WriteTopologyCreator: public simulation::Visitor
{
public:
    WriteTopologyCreator(const core::ExecParams* params);
    WriteTopologyCreator(const std::string &n, bool _writeContainers, bool _writeShellContainers, bool _createInMapping, const core::ExecParams* params, int c=0);
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n)                  { sceneName = n; }
    void setRecordContainers(bool b)                   { recordContainers=b; }
    void setRecordShellContainersV(bool b)             { recordShellContainers=b; }
    void setCreateInMapping(bool b)                    { createInMapping=b; }
    void setCounter(int c)                             { counterWriteTopology = c; }
    virtual const char* getClassName() const { return "WriteTopologyCreator"; }
protected:
    std::string sceneName;
    std::string extension;
    bool recordContainers,recordShellContainers;
    bool createInMapping;

    int counterWriteTopology; //avoid to have two same files if two Topologies are present with the same name

    void addWriteTopology(core::topology::BaseMeshTopology* topology, simulation::Node* gnode);

};



class SOFA_EXPORTER_API WriteTopologyActivator: public simulation::Visitor
{
public:
    WriteTopologyActivator( const core::ExecParams* params, bool active) : Visitor(params), state(active) {}
    virtual Result processNodeTopDown( simulation::Node*  );

    bool getState() const { return state; }
    void setState(bool active) { state=active; }
    virtual const char* getClassName() const { return "WriteTopologyActivator"; }
protected:
    void changeStateWriter(sofa::component::misc::WriteTopology *wt);

    bool state;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
