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

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/simulation/BaseSimulationExporter.h>

///////////////////////////// FORWARD DECLARATION //////////////////////////////////////////////////
namespace sofa::core
{
namespace objectmodel {
    class BaseMechanicalState;
    class Event ;
}
namespace visual {
    class VisualModel ;
}
}


////////////////////////////////// DECLARATION /////////////////////////////////////////////////////
namespace sofa::component::_stlexporter_
{

using sofa::core::behavior::BaseMechanicalState ;
using sofa::core::topology::BaseMeshTopology ;
using sofa::core::visual::VisualModel ;
using sofa::core::objectmodel::Event ;
using sofa::simulation::BaseSimulationExporter ;

class SOFA_COMPONENT_IO_MESH_API STLExporter : public BaseSimulationExporter
{
public:
    SOFA_CLASS(STLExporter, BaseSimulationExporter);

    Data<bool> d_binaryFormat;      //0 for Ascii Formats, 1 for Binary File Format
    Data<defaulttype::Vec3Types::VecCoord>               d_position; ///< points coordinates
    Data< type::vector< BaseMeshTopology::Triangle > > d_triangle; ///< triangles indices
    Data< type::vector< BaseMeshTopology::Quad > >     d_quad; ///< quads indices

    void doInit() override ;
    void doReInit() override ;
    void handleEvent(Event *) override ;

    bool write() override ;

    bool writeSTL(bool autonumbering=true);
    bool writeSTLBinary(bool autonumbering=true);

protected:
    STLExporter();
    ~STLExporter() override;

private:
    BaseMeshTopology*    m_inputtopology {nullptr};
    BaseMechanicalState* m_inputmstate   {nullptr};
    VisualModel*         m_inputvmodel   {nullptr};
};

} // namespace sofa::component::_stlexporter_

namespace sofa::component::io::mesh
{
    using STLExporter = sofa::component::_stlexporter_::STLExporter;
} // namespace sofa::component::io::mesh
