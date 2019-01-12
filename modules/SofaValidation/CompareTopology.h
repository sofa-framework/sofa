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
#ifndef SOFA_COMPONENT_MISC_COMPARETOPOLOGY_H
#define SOFA_COMPONENT_MISC_COMPARETOPOLOGY_H
#include "config.h"

#include <SofaGeneralLoader/ReadTopology.h>
#include <sofa/simulation/Visitor.h>

#include <fstream>


namespace sofa
{

namespace component
{

namespace misc
{

/** Compare Topology vectors from file at each timestep
*/
class SOFA_VALIDATION_API CompareTopology: public ReadTopology
{
public:
    SOFA_CLASS(CompareTopology,ReadTopology);
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
        if (context->getMeshTopology() == NULL)
            return false;
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
class SOFA_VALIDATION_API CompareTopologyCreator: public simulation::Visitor
{
public:
    CompareTopologyCreator(const core::ExecParams* params);
    CompareTopologyCreator(const std::string &n, const core::ExecParams* params, bool i=true, int c=0);
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n; }
    void setCounter(int c) { counterCompareTopology = c; }
    void setCreateInMapping(bool b) { createInMapping=b; }
    virtual const char* getClassName() const { return "CompareTopologyCreator"; }

protected:
    void addCompareTopology(core::topology::BaseMeshTopology* topology, simulation::Node* gnode);
    std::string sceneName;
    std::string extension;
    bool createInMapping;
    bool init;
    int counterCompareTopology; //avoid to have two same files if two Topologies are present with the same name
};


class SOFA_VALIDATION_API CompareTopologyResult: public simulation::Visitor
{
public:
    CompareTopologyResult(const core::ExecParams* params);
    virtual Result processNodeTopDown( simulation::Node*  );

    unsigned int getTotalError() {return TotalError;}
    const std::vector <unsigned int>& getErrors() {return listError;}
    unsigned int getNumCompareTopology() { return numCompareTopology; }
    virtual const char* getClassName() const { return "CompareTopologyResult"; }
protected:
    unsigned int TotalError;
    std::vector <unsigned int> listError;
    unsigned int numCompareTopology;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
