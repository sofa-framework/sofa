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
#include <sofa/component/topology/mapping/Hexa2TetraTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/component/topology/container/grid/GridTopology.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::mapping
{

using namespace sofa::defaulttype;

using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

// Register in the Factory
int Hexa2TetraTopologicalMappingClass = core::RegisterObject("Special case of mapping where HexahedronSetTopology is converted to TetrahedronSetTopology")
        .add< Hexa2TetraTopologicalMapping >()

        ;

// Implementation

Hexa2TetraTopologicalMapping::Hexa2TetraTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
    , swapping(initData(&swapping, false, "swapping","Boolean enabling to swapp hexa-edges\n in order to avoid bias effect"))
{
    m_inputType = geometry::ElementType::HEXAHEDRON;
    m_outputType = geometry::ElementType::TETRAHEDRON;
}

Hexa2TetraTopologicalMapping::~Hexa2TetraTopologicalMapping()
{
}

void Hexa2TetraTopologicalMapping::init()
{
    using namespace container::dynamic;

    if (!this->checkTopologyInputTypes()) // method will display error message if false
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    TetrahedronSetTopologyModifier* to_tstm;
    toModel->getContext()->get(to_tstm);
    if (!to_tstm)
    {
        msg_error() << "No TetrahedronSetTopologyModifier found in the Tetrahedron topology Node.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // INITIALISATION of TETRAHEDRAL mesh from HEXAHEDRAL mesh :

    TetrahedronSetTopologyContainer *to_tstc;
    toModel->getContext()->get(to_tstc);
    // Clear output topology
    to_tstc->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    const size_t nbcubes = fromModel->getNbHexahedra();

    // These values are only correct if the mesh is a grid topology
    int nx = 2;
    int ny = 1;
    //int nz = 1;
    {
        const auto* grid = dynamic_cast<container::grid::GridTopology*>(fromModel.get());
        if (grid != nullptr)
        {
            nx = grid->getNx()-1;
            ny = grid->getNy()-1;
            //nz = grid->getNz()-1;
        }
    }

    // Tesselation of each cube into 6 tetrahedra
    for (size_t i=0; i<nbcubes; i++)
    {
        core::topology::BaseMeshTopology::Hexa c = fromModel->getHexahedron(i);
#define swap(a,b) { int t = a; a = b; b = t; }
        // TODO : swap indexes where needed (currently crash in TriangleSetContainer)
        bool swapped = false;

        if(swapping.getValue())
        {
            if (!((i%nx)&1))
            {
                // swap all points on the X edges
                swap(c[0],c[1]);
                swap(c[3],c[2]);
                swap(c[4],c[5]);
                swap(c[7],c[6]);
                swapped = !swapped;
            }
            if (((i/nx)%ny)&1)
            {
                // swap all points on the Y edges
                swap(c[0],c[3]);
                swap(c[1],c[2]);
                swap(c[4],c[7]);
                swap(c[5],c[6]);
                swapped = !swapped;
            }
            if ((i/(nx*ny))&1)
            {
                // swap all points on the Z edges
                swap(c[0],c[4]);
                swap(c[1],c[5]);
                swap(c[2],c[6]);
                swap(c[3],c[7]);
                swapped = !swapped;
            }
        }
#undef swap
        if(!swapped)
        {
            to_tstc->addTetra(c[0],c[5],c[1],c[6]);
            to_tstc->addTetra(c[0],c[1],c[3],c[6]);
            to_tstc->addTetra(c[1],c[3],c[6],c[2]);
            to_tstc->addTetra(c[6],c[3],c[0],c[7]);
            to_tstc->addTetra(c[6],c[7],c[0],c[5]);
            to_tstc->addTetra(c[7],c[5],c[4],c[0]);
        }
        else
        {
            to_tstc->addTetra(c[0],c[5],c[6],c[1]);
            to_tstc->addTetra(c[0],c[1],c[6],c[3]);
            to_tstc->addTetra(c[1],c[3],c[2],c[6]);
            to_tstc->addTetra(c[6],c[3],c[7],c[0]);
            to_tstc->addTetra(c[6],c[7],c[5],c[0]);
            to_tstc->addTetra(c[7],c[5],c[0],c[4]);
        }
        for(int j=0; j<6; j++)
            Loc2GlobVec.push_back(i);
        Glob2LocMap[i] = (unsigned int)Loc2GlobVec.size()-1;
    }

    // Need to fully init the target topology
    toModel->init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

Index Hexa2TetraTopologicalMapping::getFromIndex(Index /*ind*/)
{

    return sofa::InvalidID;
}

void Hexa2TetraTopologicalMapping::updateTopologicalMappingTopDown()
{
    msg_warning() << "Method Hexa2TetraTopologicalMapping::updateTopologicalMappingTopDown() not yet implemented!";
// TODO...
}


} //namespace sofa::component::topology::mapping
