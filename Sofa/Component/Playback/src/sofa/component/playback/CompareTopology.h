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

#include <sofa/component/playback/ReadTopology.h>
#include <sofa/simulation/Visitor.h>

#include <fstream>


namespace sofa::component::playback
{

/** Compare Topology vectors from file at each timestep
*/
class SOFA_COMPONENT_PLAYBACK_API CompareTopology: public ReadTopology
{
public:
    SOFA_CLASS(CompareTopology, playback::ReadTopology);
protected:
    /** Default constructor
    */
    CompareTopology();
public:
    void handleEvent(sofa::core::objectmodel::Event* event) override;

    /// Compute the total number of errors
    void processCompareTopology();

    /** Pre-construction check method called by ObjectFactory.
    Check that DataTypes matches the MechanicalTopology.*/
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMeshTopology() == nullptr)
        {
            arg->logError("Cannot find a mesh topology in the current context");
            return false;
        }
        return BaseObject::canCreate(obj, context, arg);
    }

    /// Return the total number of errors
    unsigned int getTotalError() {return TotalError;}
    /// Return the errors by containers
    const std::vector <unsigned int>& getErrors() {return listError;}



protected :

    /// Number of error by container
    unsigned int EdgesError;
    unsigned int TrianglesError;
    unsigned int QuadsError;
    unsigned int TetrahedraError;
    unsigned int HexahedraError;

    unsigned int TotalError;

    std::vector <unsigned int> listError;
};


/// Create CompareTopology component in the graph each time needed
class SOFA_COMPONENT_PLAYBACK_API CompareTopologyCreator: public simulation::Visitor
{
public:
    CompareTopologyCreator(const core::ExecParams* params);
    CompareTopologyCreator(const std::string &n, const core::ExecParams* params, bool i=true, int c=0);
    Result processNodeTopDown( simulation::Node*  ) override;

    void setSceneName(std::string &n) { sceneName = n; }
    void setCounter(int c) { counterCompareTopology = c; }
    void setCreateInMapping(bool b) { createInMapping=b; }
    const char* getClassName() const override { return "CompareTopologyCreator"; }

protected:
    void addCompareTopology(core::topology::BaseMeshTopology* topology, simulation::Node* gnode);
    std::string sceneName;
    std::string extension;
    bool createInMapping;
    bool init;
    int counterCompareTopology; //avoid to have two same files if two Topologies are present with the same name
};


class SOFA_COMPONENT_PLAYBACK_API CompareTopologyResult: public simulation::Visitor
{
public:
    CompareTopologyResult(const core::ExecParams* params);
    Result processNodeTopDown( simulation::Node*  ) override;

    unsigned int getTotalError() {return TotalError;}
    const std::vector <unsigned int>& getErrors() {return listError;}
    unsigned int getNumCompareTopology() { return numCompareTopology; }
    const char* getClassName() const override { return "CompareTopologyResult"; }
protected:
    unsigned int TotalError;
    std::vector <unsigned int> listError;
    unsigned int numCompareTopology;
};

} // namespace sofa::component::playback
