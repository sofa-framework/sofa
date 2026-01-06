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
#include <sofa/component/io/mesh/config.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

#include <fstream>

#include <sofa/simulation/BaseSimulationExporter.h>


namespace sofa::component::_vtkexporter_
{

class SOFA_COMPONENT_IO_MESH_API VTKExporter : public sofa::simulation::BaseSimulationExporter
{
public:
    SOFA_CLASS(VTKExporter,sofa::simulation::BaseSimulationExporter);

protected:
    SingleLink<VTKExporter, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STRONGLINK> m_topology;
    SingleLink<VTKExporter, sofa::core::behavior::BaseMechanicalState, BaseLink::FLAG_STRONGLINK> m_mstate;

    std::unique_ptr<std::ofstream> m_outfile;

    void fetchDataFields(const type::vector<std::string>& strData, type::vector<std::string>& objects, type::vector<std::string>& fields, type::vector<std::string>& names);
    bool write() override;
    bool writeVTKSimple();
    bool writeVTKXML();
    void writeParallelFile();
    void writeData(const type::vector<std::string>& objects, const type::vector<std::string>& fields, const type::vector<std::string>& names);
    void writeDataArray(const type::vector<std::string>& objects, const type::vector<std::string>& fields, const type::vector<std::string>& names);
    std::string segmentString(std::string str, unsigned int n);

public:
    Data<bool> d_fileFormat; ///< Set to true to use XML format
    Data<defaulttype::Vec3Types::VecCoord> d_position; ///< points position (will use points from topology or mechanical state if this is empty)
    Data<bool> d_writeEdges; ///< write edge topology
    Data<bool> d_writeTriangles; ///< write triangle topology
    Data<bool> d_writeQuads; ///< write quad topology
    Data<bool> d_writeTetras; ///< write tetra topology
    Data<bool> d_writeHexas; ///< write hexa topology
    Data<type::vector<std::string> > d_dPointsDataFields; ///< Data to visualize (on points)
    Data<type::vector<std::string> > d_dCellsDataFields; ///< Data to visualize (on cells)
    Data<bool> d_overwrite; ///< overwrite the file, otherwise create a new file at each export, with suffix in the filename

    int nbFiles;

    type::vector<std::string> pointsDataObject;
    type::vector<std::string> pointsDataField;
    type::vector<std::string> pointsDataName;

    type::vector<std::string> cellsDataObject;
    type::vector<std::string> cellsDataField;
    type::vector<std::string> cellsDataName;
protected:
    VTKExporter();
    ~VTKExporter() override;
public:
    void doInit() override;
    void doReInit() override;
    void handleEvent(sofa::core::objectmodel::Event *) override;
};

} // namespace sofa::component::_vtkexporter_

namespace sofa::component::io::mesh 
{
    using VTKExporter = _vtkexporter_::VTKExporter;
} // namespace sofa::component::io::mesh 
