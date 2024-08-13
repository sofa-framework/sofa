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

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Visitor.h>

#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
#include <zlib.h>
#endif

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <fstream>

namespace sofa::component::playback
{

/** Read Topology containers informations from file at each timestep
*/
class SOFA_COMPONENT_PLAYBACK_API ReadTopology: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(ReadTopology,core::objectmodel::BaseObject);

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    sofa::core::objectmodel::DataFileName f_filename;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data < double > f_interval;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data < double > f_shift;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data < bool > f_loop;

    sofa::core::objectmodel::DataFileName d_filename;
    Data < double > d_interval; ///< time duration between inputs
    Data < double > d_shift; ///< shift between times in the file and times when they will be read
    Data < bool > d_loop; ///< set to 'true' to re-read the file when reaching the end

    /// Link to be set to the topology container in the component graph.
    SingleLink <ReadTopology, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    core::topology::BaseMeshTopology* m_topology;
    std::ifstream* infile;
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    gzFile gzfile;
#endif
    double nextTime;
    double lastTime;
    double loopTime;

    ReadTopology();

    ~ReadTopology() override;
public:
    void init() override;

    void reset() override;

    void setTime(double time);

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void processReadTopology();
    void processReadTopology(double time);

    /// Read the next values in the file corresponding to the last timestep before the given time
    bool readNext(double time, std::vector<std::string>& lines);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T* obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMeshTopologyLink() == nullptr) {
            arg->logError("No mesh topology found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }



    // Tab of 2D elements composition
    Data< type::vector< type::fixed_array <unsigned int,2> > > edges;
    Data< type::vector< type::fixed_array <unsigned int,3> > > triangles;
    Data< type::vector< type::fixed_array <unsigned int,4> > > quads;

    // Tab of 3D elements composition
    Data< type::vector< type::fixed_array<unsigned int,4> > > tetrahedra;
    Data< type::vector< type::fixed_array<unsigned int,8> > > hexahedra;


};


///Create ReadTopology component in the graph each time needed
class SOFA_COMPONENT_PLAYBACK_API ReadTopologyCreator: public simulation::Visitor
{
public:
    ReadTopologyCreator(const core::ExecParams* params);
    ReadTopologyCreator(const std::string &n, bool _createInMapping, const core::ExecParams* params, bool i=true, int c=0 );
    Result processNodeTopDown( simulation::Node*  ) override;

    void setSceneName(std::string &n) { sceneName = n;}
    void setCounter(int c) {counterReadTopology = c;}
    void setCreateInMapping(bool b) {createInMapping=b;}
    const char* getClassName() const override { return "ReadTopologyCreator"; }
protected:
    std::string sceneName;
    std::string extension;
    bool createInMapping;
    bool init;
    int counterReadTopology; //avoid to have two same files if two Topologies are present with the same name

    void addReadTopology(core::topology::BaseMeshTopology* topology, simulation::Node* gnode);

};

class SOFA_COMPONENT_PLAYBACK_API ReadTopologyActivator: public simulation::Visitor
{
public:
    ReadTopologyActivator(const core::ExecParams* params, bool active)
        :Visitor(params), state(active) {}
    Result processNodeTopDown( simulation::Node*  ) override;

    bool getTopology() const {return state;}
    void setTopology(bool active) {state=active;}
    const char* getClassName() const override { return "ReadTopologyActivator"; }
protected:
    void changeTopologyReader(sofa::component::playback::ReadTopology *rt);

    bool state;
};

class SOFA_COMPONENT_PLAYBACK_API ReadTopologyModifier: public simulation::Visitor
{
public:
    ReadTopologyModifier(const core::ExecParams* params, double _time)
        :Visitor(params), time(_time) {}

    Result processNodeTopDown( simulation::Node*  ) override;

    double getTime() const { return time; }
    void setTime(double _time) { time=_time; }
    const char* getClassName() const override { return "ReadTopologyModifier"; }
protected:
    void changeTimeReader(sofa::component::playback::ReadTopology *rt) { rt->processReadTopology(time); }

    double time;
};

} // namespace sofa::component::playback
