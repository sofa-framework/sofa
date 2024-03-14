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
#include <sofa/component/mapping/linear/SimpleTesselatedHexaTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>


#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear
{
using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::component::mapping::linear;
using namespace sofa::core::topology;

// Register in the Factory
int SimpleTesselatedHexaTopologicalMappingClass = core::RegisterObject ( "Special case of mapping where HexahedronSetTopology is converted into a finer HexahedronSetTopology" )
        .add< SimpleTesselatedHexaTopologicalMapping >()
        ;

// Implementation
SimpleTesselatedHexaTopologicalMapping::SimpleTesselatedHexaTopologicalMapping()
    : sofa::core::topology::TopologicalMapping()
{
    m_inputType = geometry::ElementType::HEXAHEDRON;
    m_outputType = geometry::ElementType::HEXAHEDRON;
}

void SimpleTesselatedHexaTopologicalMapping::init()
{
    // Check input/output topology
    if (!this->checkTopologyInputTypes()) // method will display error message if false
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    toModel->clear();

    for (std::size_t i=0; i<fromModel->getNbPoints(); ++i)
    {
        // points mapped from points
        pointMappedFromPoint.push_back(i);
        toModel->addPoint(fromModel->getPX(i), fromModel->getPY(i), fromModel->getPZ(i));
    }

    sofa::Index pointIndex = static_cast<sofa::Index>(pointMappedFromPoint.size());
    Vec3 p;

    for (std::size_t i=0; i<fromModel->getNbHexahedra(); ++i)
    {
        core::topology::BaseMeshTopology::Hexa h = fromModel->getHexahedron(i);

        Vec3 p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
        Vec3 p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
        Vec3 p2(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
        Vec3 p3(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
        Vec3 p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
        Vec3 p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
        Vec3 p6(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));
        Vec3 p7(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));

        // points mapped from edges
        bool insertResultSuccessful = pointMappedFromEdge.insert({ {h[0],h[1]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p0+p1)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[1],h[(2)]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p1+p2)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[3],h[2]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p3+p2)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[0],h[3]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p0+p3)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[0],h[4]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p0+p4)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[1],h[5]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p1+p5)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[2],h[6]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p2+p6)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[3],h[7]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p3+p7)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[4],h[5]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p4+p5)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[5],h[6]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p5+p6)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[7],h[6]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p7+p6)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertResultSuccessful = pointMappedFromEdge.insert({{h[4],h[7]},pointIndex }).second;
        if(insertResultSuccessful)
        {
            p = (p4+p7)/2;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        // points mapped from facets
        bool insertFacetsResultSuccessful = pointMappedFromFacet.insert({ {h[0], h[1], h[2], h[3]}, pointIndex }).second;
        if (insertFacetsResultSuccessful)
        {
            p = (p0+p1+p2+p3)/4;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertFacetsResultSuccessful = pointMappedFromFacet.insert({ {h[0], h[1], h[5], h[4] }, pointIndex }).second;
        if (insertFacetsResultSuccessful)
        {
            p = (p0+p1+p5+p4)/4;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertFacetsResultSuccessful = pointMappedFromFacet.insert({ {h[1], h[2], h[6], h[5]}, pointIndex }).second;
        if (insertFacetsResultSuccessful)
        {
            p = (p1+p2+p6+p5)/4;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertFacetsResultSuccessful = pointMappedFromFacet.insert({ {h[3], h[2], h[6], h[7]}, pointIndex }).second;
        if (insertFacetsResultSuccessful)
        {
            p = (p3+p2+p6+p7)/4;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertFacetsResultSuccessful = pointMappedFromFacet.insert({ {h[0], h[3], h[7], h[4]}, pointIndex }).second;
        if (insertFacetsResultSuccessful)
        {
            p = (p0+p4+p7+p3)/4;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        insertFacetsResultSuccessful = pointMappedFromFacet.insert({ {h[4], h[5], h[6], h[7]}, pointIndex }).second;
        if (insertFacetsResultSuccessful)
        {
            p = (p4+p5+p6+p7)/4;
            toModel->addPoint(p[0], p[1], p[2]);
            pointIndex++;
        }

        // points mapped from hexahedra
        pointMappedFromHexa.push_back((int)pointIndex);
        p = (p0+p1+p2+p3+p4+p5+p6+p7)/8;
        toModel->addPoint(p[0], p[1], p[2]);
        pointIndex++;
    }

    for (unsigned int i=0; i<fromModel->getNbHexahedra(); ++i)
    {
        core::topology::BaseMeshTopology::Hexa h = fromModel->getHexahedron(i);

        Vec3d p0(fromModel->getPX(h[0]), fromModel->getPY(h[0]), fromModel->getPZ(h[0]));
        Vec3d p1(fromModel->getPX(h[1]), fromModel->getPY(h[1]), fromModel->getPZ(h[1]));
        Vec3d p2(fromModel->getPX(h[2]), fromModel->getPY(h[2]), fromModel->getPZ(h[2]));
        Vec3d p3(fromModel->getPX(h[3]), fromModel->getPY(h[3]), fromModel->getPZ(h[3]));
        Vec3d p4(fromModel->getPX(h[4]), fromModel->getPY(h[4]), fromModel->getPZ(h[4]));
        Vec3d p5(fromModel->getPX(h[5]), fromModel->getPY(h[5]), fromModel->getPZ(h[5]));
        Vec3d p6(fromModel->getPX(h[6]), fromModel->getPY(h[6]), fromModel->getPZ(h[6]));
        Vec3d p7(fromModel->getPX(h[7]), fromModel->getPY(h[7]), fromModel->getPZ(h[7]));

        toModel->addHexa(h[0],
                pointMappedFromEdge[{h[0],h[1]}],
                pointMappedFromFacet[{h[0],h[1],h[2],h[3]}],
                pointMappedFromEdge[{h[0],h[3]}],
                pointMappedFromEdge[{h[0],h[4]}],
                pointMappedFromFacet[{h[0],h[1],h[5],h[4]}],
                pointMappedFromHexa[i],
                pointMappedFromFacet[{h[0],h[3],h[7],h[4]}]);

        toModel->addHexa(pointMappedFromEdge[{h[0],h[1]}],
                h[1],
                pointMappedFromEdge[{h[1],h[2]}],
                pointMappedFromFacet[{h[0],h[1],h[2],h[3]}],
                pointMappedFromFacet[{h[0],h[1],h[5],h[4]}],
                pointMappedFromEdge[{h[1],h[5]}],
                pointMappedFromFacet[{h[1],h[2],h[6],h[5]}],
                pointMappedFromHexa[i]);

        toModel->addHexa(pointMappedFromFacet[{h[0],h[1],h[2],h[3]}],
                pointMappedFromEdge[{h[1],h[2]}],
                h[2],
                pointMappedFromEdge[{h[3],h[2]}],
                pointMappedFromHexa[i],
                pointMappedFromFacet[{h[1],h[2],h[6],h[5]}],
                pointMappedFromEdge[{h[2],h[6]}],
                pointMappedFromFacet[{h[3],h[2],h[6],h[7]}]);

        toModel->addHexa(pointMappedFromEdge[{h[0],h[3]}],
                pointMappedFromFacet[{h[0],h[1],h[2],h[3]}],
                pointMappedFromEdge[{h[3],h[2]}],
                h[3],
                pointMappedFromFacet[{h[0],h[3],h[7],h[4]}],
                pointMappedFromHexa[i],
                pointMappedFromFacet[{h[3],h[2],h[6],h[7]}],
                pointMappedFromEdge[{h[3],h[7]}]);

        toModel->addHexa(pointMappedFromEdge[{h[0],h[4]}],
                pointMappedFromFacet[{h[0],h[1],h[5],h[4]}],
                pointMappedFromHexa[i],
                pointMappedFromFacet[{h[0],h[3],h[7],h[4]}],
                h[4],
                pointMappedFromEdge[{h[4],h[5]}],
                pointMappedFromFacet[{h[4],h[5],h[6],h[7]}],
                pointMappedFromEdge[{h[4],h[7]}]);

        toModel->addHexa(pointMappedFromFacet[{h[0],h[1],h[5],h[4]}],
                pointMappedFromEdge[{h[1],h[5]}],
                pointMappedFromFacet[{h[1],h[2],h[6],h[5]}],
                pointMappedFromHexa[i],
                pointMappedFromEdge[{h[4],h[5]}],
                h[5],
                pointMappedFromEdge[{h[5],h[6]}],
                pointMappedFromFacet[{h[4],h[5],h[6],h[7]}]);

        toModel->addHexa(pointMappedFromHexa[i],
                pointMappedFromFacet[{h[1],h[2],h[6],h[5]}],
                pointMappedFromEdge[{h[2],h[6]}],
                pointMappedFromFacet[{h[3],h[2],h[6],h[7]}],
                pointMappedFromFacet[{h[4],h[5],h[6],h[7]}],
                pointMappedFromEdge[{h[5],h[6]}],
                h[6],
                pointMappedFromEdge[{h[7],h[6]}]);

        toModel->addHexa(pointMappedFromFacet[{h[0],h[3],h[7],h[4]}],
                pointMappedFromHexa[i],
                pointMappedFromFacet[{h[3],h[2],h[6],h[7]}],
                pointMappedFromEdge[{h[3],h[7]}],
                pointMappedFromEdge[{h[4],h[7]}],
                pointMappedFromFacet[{h[4],h[5],h[6],h[7]}],
                pointMappedFromEdge[{h[7],h[6]}],
                h[7]);
    }

    // Need to fully init the target topology
    toModel->init();
}

} //namespace sofa::component::mapping::linear
