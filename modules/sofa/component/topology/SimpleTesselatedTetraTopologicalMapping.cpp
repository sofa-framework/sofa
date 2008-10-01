/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/topology/SimpleTesselatedTetraTopologicalMapping.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>

#include <sofa/component/topology/PointSetTopologyChange.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::component::topology;
using namespace sofa::core::componentmodel::topology;

SOFA_DECL_CLASS ( SimpleTesselatedTetraTopologicalMapping )

// Register in the Factory
int SimpleTesselatedTetraTopologicalMappingClass = core::RegisterObject ( "Special case of mapping where TetrahedronSetTopology is converted into a finer TetrahedronSetTopology" )
        .add< SimpleTesselatedTetraTopologicalMapping >()
        ;

// Implementation
SimpleTesselatedTetraTopologicalMapping::SimpleTesselatedTetraTopologicalMapping ( In* from, Out* to )
    : TopologicalMapping ( from, to ),
      object1 ( initData ( &object1, std::string ( "../.." ), "object1", "First object to map" ) ),
      object2 ( initData ( &object2, std::string ( ".." ), "object2", "Second object to map" ) )
{
}

void SimpleTesselatedTetraTopologicalMapping::init()
{
    if(fromModel)
    {
        TetrahedronSetTopologyContainer *from_tstc;
        fromModel->getContext()->get(from_tstc);
        if(toModel)
        {
            TetrahedronSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);
            to_tstc->clear();
            if(!from_tstc->hasPos())
            {
//				MeshLoader *mshLoader;
//				fromModel->getContext()->get(mshLoader);
//				from_tstc->loadFromMeshLoader(mshLoader);
            }

            std::cout << from_tstc->getNbPoints() << std::endl;

            pointSource.resize(from_tstc->getNbPoints()+from_tstc->getNbEdges());


            for (int i=0; i<from_tstc->getNbPoints(); i++)
            {
                to_tstc->addPoint(from_tstc->getPX(i), from_tstc->getPY(i), from_tstc->getPZ(i));
                std::cout << from_tstc->getPX(i) << " " << from_tstc->getPY(i) << " " << from_tstc->getPZ(i) << std::endl;

                pointMappedFromPoint.push_back(i);
                pointSource[i] = i;
            }

            int newPointIndex = to_tstc->getNbPoints();

            for (int i=0; i<from_tstc->getNbEdges(); i++)
            {
                Edge e = from_tstc->getEdge(i);

                to_tstc->addPoint(
                    (from_tstc->getPX(e[0]) + from_tstc->getPX(e[1]))/2,
                    (from_tstc->getPY(e[0]) + from_tstc->getPY(e[1]))/2,
                    (from_tstc->getPZ(e[0]) + from_tstc->getPZ(e[1]))/2
                );

                pointMappedFromEdge.push_back(newPointIndex); newPointIndex++;
                pointSource[newPointIndex] = -1-i;
            }

            fixed_array <int, 8> newTetrasIndices;
            int newTetraIndex = to_tstc->getNbTetras();

            tetraSource.resize(8*from_tstc->getNbTetras());

            for (int i=0; i<from_tstc->getNbTetras(); i++)
            {
                Tetra t = from_tstc->getTetra(i);
                TetraEdges e = from_tstc->getEdgeTetraShell(i);
                to_tstc->addTetra(t[0],	pointMappedFromEdge[e[0]], pointMappedFromEdge[e[1]], pointMappedFromEdge[e[2]]);
                newTetrasIndices[0] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(t[1],	pointMappedFromEdge[e[0]], pointMappedFromEdge[e[4]], pointMappedFromEdge[e[3]]);
                newTetrasIndices[1] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(t[2],	pointMappedFromEdge[e[1]], pointMappedFromEdge[e[3]], pointMappedFromEdge[e[5]]);
                newTetrasIndices[2] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(t[3],	pointMappedFromEdge[e[2]], pointMappedFromEdge[e[5]], pointMappedFromEdge[e[4]]);
                newTetrasIndices[3] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdge[e[1]], pointMappedFromEdge[e[0]], pointMappedFromEdge[e[3]], pointMappedFromEdge[e[5]]);
                newTetrasIndices[4] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdge[e[1]], pointMappedFromEdge[e[0]], pointMappedFromEdge[e[5]], pointMappedFromEdge[e[2]]);
                newTetrasIndices[5] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdge[e[4]], pointMappedFromEdge[e[0]], pointMappedFromEdge[e[5]], pointMappedFromEdge[e[3]]);
                newTetrasIndices[6] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdge[e[4]], pointMappedFromEdge[e[0]], pointMappedFromEdge[e[2]], pointMappedFromEdge[e[5]]);
                newTetrasIndices[7] = newTetraIndex; tetraSource[newTetraIndex] = i; newTetraIndex++;

                tetrasMappedFromTetra.push_back(newTetrasIndices);
            }
            toModel->init();
        }
    }
}

void SimpleTesselatedTetraTopologicalMapping::updateTopologicalMappingBottomUp()
{
    if(fromModel)
    {
        TetrahedronSetTopologyContainer *from_tstc;
        fromModel->getContext()->get(from_tstc);
        TriangleSetTopologyModifier *from_tstm;
        fromModel->getContext()->get(from_tstm);
        if(toModel)
        {
            TetrahedronSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);

            std::list<const TopologyChange *>::const_iterator changeIt=toModel->firstChange();
            std::list<const TopologyChange *>::const_iterator itEnd=toModel->lastChange();

            while( changeIt != itEnd )
            {
                TopologyChangeType changeType = (*changeIt)->getChangeType();

                switch( changeType )
                {
                case core::componentmodel::topology::POINTSINDICESSWAP:
                {
                    unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
                    unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];
                    // i1 and i2 swapped in output model
                    swapOutputPoints(i1,i2);
                    break;
                }
                case core::componentmodel::topology::POINTSADDED:
                {
                    /// @TODO
                    break;
                }
                case core::componentmodel::topology::POINTSREMOVED:
                {
                    const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();
                    removeOutputPoints( tab );
                    break;
                }
                case core::componentmodel::topology::POINTSRENUMBERING:
                {
                    const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getinv_IndexArray();
                    renumberOutputPoints( tab );
                    break;
                }
                case core::componentmodel::topology::TETRAHEDRAADDED:
                {
                    /// @TODO
                    break;
                }
                case core::componentmodel::topology::TETRAHEDRAREMOVED:
                {
                    const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();
                    removeOutputTetras( tab );
                    break;
                }
                default: break;

                }
                ++changeIt;
            }
        }
    }

}

void SimpleTesselatedTetraTopologicalMapping::swapOutputPoints(int i1, int i2)
{
    // first update pointSource
    int i1Source = pointSource[i1];
    int i2Source = pointSource[i2];
    setPointSource(i1, i2Source);
    setPointSource(i2, i1Source);
}

void SimpleTesselatedTetraTopologicalMapping::removeOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    unsigned int last = pointSource.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputPoints( index[i], last );
        int source = pointSource[last];
        if (source < 0)
        {
            pointMappedFromEdge[1-source] = -1;
        }
        else
        {
            pointMappedFromPoint[source] = -1;
        }
        --last;
    }

    pointSource.resize( pointSource.size() - index.size() );
}

void SimpleTesselatedTetraTopologicalMapping::renumberOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    helper::vector<int> copy = pointSource;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        setPointSource(i, copy[ index[i] ]);
    }
}


void SimpleTesselatedTetraTopologicalMapping::swapOutputTetras(int i1, int i2)
{
    // first update pointSource
    int i1Source = tetraSource[i1];
    int i2Source = tetraSource[i2];
    tetraSource[i1] = i2Source;
    tetraSource[i2] = i1Source;
    for (int j=0; j<8; ++j)
        if (tetrasMappedFromTetra[i1Source][j] == i1)
        {
            tetrasMappedFromTetra[i1Source][j] = i2;
            break;
        }
    for (int j=0; j<8; ++j)
        if (tetrasMappedFromTetra[i2Source][j] == i2)
        {
            tetrasMappedFromTetra[i2Source][j] = i1;
            break;
        }
}

void SimpleTesselatedTetraTopologicalMapping::removeOutputTetras( const sofa::helper::vector<unsigned int>& index )
{
    int last = tetraSource.size() -1;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputTetras( index[i], last );
        int source = tetraSource[last];
        int nbt = 0;
        for (int j=0; j<8; ++j)
            if (tetrasMappedFromTetra[source][j] == last)
            {
                tetrasMappedFromTetra[source][j] = -1;
            }
            else if (tetrasMappedFromTetra[source][j] != -1)
                ++nbt;
        if (nbt == 0) // we need to remove the source tetra
        {
            std::cout << "SimpleTesselatedTetraTopologicalMapping: source tetra "<<source<<" needs to be removed."<<std::endl;
            tetrasToRemove.push_back(source);
        }
        --last;
    }

    tetraSource.resize( tetraSource.size() - index.size() );

}

} // namespace topology
} // namespace component
} // namespace sofa

