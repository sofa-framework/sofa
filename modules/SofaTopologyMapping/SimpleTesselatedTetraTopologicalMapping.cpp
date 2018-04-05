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
#include <SofaTopologyMapping/SimpleTesselatedTetraTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>

#include <SofaBaseTopology/TopologyData.inl>


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
using namespace sofa::core::topology;
using sofa::helper::fixed_array;

SOFA_DECL_CLASS ( SimpleTesselatedTetraTopologicalMapping )

// Register in the Factory
int SimpleTesselatedTetraTopologicalMappingClass = core::RegisterObject ( "Special case of mapping where TetrahedronSetTopology is converted into a finer TetrahedronSetTopology" )
        .add< SimpleTesselatedTetraTopologicalMapping >()
        ;

// Implementation
SimpleTesselatedTetraTopologicalMapping::SimpleTesselatedTetraTopologicalMapping ()
    : tetrahedraMappedFromTetra( initData ( &tetrahedraMappedFromTetra, "tetrahedraMappedFromTetra", "Each Tetrahedron of the input topology is mapped to the 8 tetrahedrons in which it can be divided")),
      tetraSource( initData ( &tetraSource, "tetraSource", "Which tetra from the input topology map to a given tetra in the output topology (-1 if none)")),
      d_pointMappedFromPoint( initData ( &d_pointMappedFromPoint, "pointMappedFromPoint", "Each point of the input topology is mapped to the same point")),
      d_pointMappedFromEdge( initData ( &d_pointMappedFromEdge, "pointMappedFromEdge", "Each edge of the input topology is mapped to his midpoint")),
      d_pointSource( initData ( &d_pointSource, "pointSource", "Which input topology element map to a given point in the output topology : 0 -> none, > 0 -> point index + 1, < 0 , - edge index -1"))
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

            helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
            helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;
            helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromEdgeData = d_pointMappedFromEdge;

            sofa::helper::vector<helper::fixed_array<int, 8> >& tetrahedraMappedFromTetraData = *(tetrahedraMappedFromTetra.beginEdit());
            helper::vector<int>& tetraSourceData = *(tetraSource.beginEdit());

            TetrahedronSetTopologyContainer *to_tstc;
            toModel->getContext()->get(to_tstc);
            to_tstc->clear();
            if(!from_tstc->hasPos())
            {
//				MeshLoader *mshLoader;
//				fromModel->getContext()->get(mshLoader);
//				from_tstc->loadFromMeshLoader(mshLoader);
            }

            //sout << from_tstc->getNbPoints() << sendl;

            pointSourceData.resize(from_tstc->getNbPoints()+from_tstc->getNbEdges());


            for (int i=0; i<from_tstc->getNbPoints(); i++)
            {
                to_tstc->addPoint(from_tstc->getPX(i), from_tstc->getPY(i), from_tstc->getPZ(i));
                //sout << from_tstc->getPX(i) << " " << from_tstc->getPY(i) << " " << from_tstc->getPZ(i) << sendl;

                pointMappedFromPointData.push_back(i);
                pointSourceData[i] = i+1;
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

                pointMappedFromEdgeData.push_back(newPointIndex);
                pointSourceData[newPointIndex] = -(i+1);
                newPointIndex++;
            }

            fixed_array <int, 8> newTetrahedraIndices;
            int newTetraIndex = to_tstc->getNbTetrahedra();

            tetraSourceData.resize(8*from_tstc->getNbTetrahedra());

            for (int i=0; i<from_tstc->getNbTetrahedra(); i++)
            {
                core::topology::BaseMeshTopology::Tetra t = from_tstc->getTetrahedron(i);
                core::topology::BaseMeshTopology::EdgesInTetrahedron e = from_tstc->getEdgesInTetrahedron(i);
                to_tstc->addTetra(t[0],	pointMappedFromEdgeData[e[0]], pointMappedFromEdgeData[e[1]], pointMappedFromEdgeData[e[2]]);
                newTetrahedraIndices[0] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(t[1],	pointMappedFromEdgeData[e[0]], pointMappedFromEdgeData[e[4]], pointMappedFromEdgeData[e[3]]);
                newTetrahedraIndices[1] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(t[2],	pointMappedFromEdgeData[e[1]], pointMappedFromEdgeData[e[3]], pointMappedFromEdgeData[e[5]]);
                newTetrahedraIndices[2] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(t[3],	pointMappedFromEdgeData[e[2]], pointMappedFromEdgeData[e[5]], pointMappedFromEdgeData[e[4]]);
                newTetrahedraIndices[3] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdgeData[e[1]], pointMappedFromEdgeData[e[0]], pointMappedFromEdgeData[e[3]], pointMappedFromEdgeData[e[5]]);
                newTetrahedraIndices[4] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdgeData[e[1]], pointMappedFromEdgeData[e[0]], pointMappedFromEdgeData[e[5]], pointMappedFromEdgeData[e[2]]);
                newTetrahedraIndices[5] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdgeData[e[4]], pointMappedFromEdgeData[e[0]], pointMappedFromEdgeData[e[5]], pointMappedFromEdgeData[e[3]]);
                newTetrahedraIndices[6] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                to_tstc->addTetra(pointMappedFromEdgeData[e[4]], pointMappedFromEdgeData[e[0]], pointMappedFromEdgeData[e[2]], pointMappedFromEdgeData[e[5]]);
                newTetrahedraIndices[7] = newTetraIndex; tetraSourceData[newTetraIndex] = i; newTetraIndex++;

                tetrahedraMappedFromTetraData.push_back(newTetrahedraIndices);
            }
            toModel->init();
            tetrahedraMappedFromTetra.endEdit();
        }
    }
}

void SimpleTesselatedTetraTopologicalMapping::updateTopologicalMappingBottomUp()
{
    if(fromModel && toModel)
    {
        //TetrahedronSetTopologyContainer *from_tstc;
        //fromModel->getContext()->get(from_tstc);
        TetrahedronSetTopologyModifier *from_tstm;
        fromModel->getContext()->get(from_tstm);
        //TetrahedronSetTopologyContainer *to_tstc;
        //toModel->getContext()->get(to_tstc);

        std::list<const TopologyChange *>::const_iterator changeIt=toModel->beginChange();
        std::list<const TopologyChange *>::const_iterator itEnd=toModel->endChange();

        while( changeIt != itEnd )
        {
            TopologyChangeType changeType = (*changeIt)->getChangeType();

            switch( changeType )
            {
            case core::topology::POINTSINDICESSWAP:
            {
                unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
                unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];
                // i1 and i2 swapped in output model
//				 sout << "OUTPUT SWAP POINTS "<<i1 << " " << i2 << sendl;
                swapOutputPoints(i1,i2);
                break;
            }
            case core::topology::POINTSADDED:
            {
                /// @todo
                break;
            }
            case core::topology::POINTSREMOVED:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();
//				 sout << "OUTPUT REMOVE POINTS "<<tab << sendl;
                removeOutputPoints( tab );
                break;
            }
            case core::topology::POINTSRENUMBERING:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getinv_IndexArray();
//				 sout << "OUTPUT RENUMBER POINTS "<<tab << sendl;
                renumberOutputPoints( tab );
                break;
            }
            case core::topology::TETRAHEDRAADDED:
            {
                /// @todo
                break;
            }
            case core::topology::TETRAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();
//				sout << "OUTPUT REMOVE TETRAHEDRA "<<tab << sendl;
                removeOutputTetrahedra( tab );
                break;
            }
            case core::topology::ENDING_EVENT:
            {
                //sout << "(*pointMappedFromPointData) = " << (*pointMappedFromPointData)<<sendl;
                //sout << "pointMappedFromEdge = " << pointMappedFromEdge<<sendl;
                //sout << "(*pointSourceData) = " << pointMappedFromEdge<<sendl;
                if (from_tstm != NULL && !tetrahedraToRemove.empty())
                {
                    sofa::helper::vector<unsigned int> vitems;
                    vitems.reserve(tetrahedraToRemove.size());
                    vitems.insert(vitems.end(), tetrahedraToRemove.rbegin(), tetrahedraToRemove.rend());

                    from_tstm->removeItems(vitems);
                    //from_tstm->removeTetrahedraWarning(vitems);
                    //from_tstm->propagateTopologicalChanges();
                    //from_tstm->removeTetrahedraProcess(vitems);

                    tetrahedraToRemove.clear();

                    from_tstm->propagateTopologicalChanges();
                    from_tstm->notifyEndingEvent();
                    from_tstm->propagateTopologicalChanges();
                }

                break;
            }
            default: break;

            }
            ++changeIt;
        }
    }
}

void SimpleTesselatedTetraTopologicalMapping::swapOutputPoints(int i1, int i2)
{
    helper::ReadAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;

    // first update pointSourceData
    int i1Source = pointSourceData[i1];
    int i2Source = pointSourceData[i2];
//    sout << "swap output points "<<i1 << " " << i2 << " from source " << i1Source << " " << i2Source << sendl;
    setPointSource(i1, i2Source);
    setPointSource(i2, i1Source);

}

void SimpleTesselatedTetraTopologicalMapping::removeOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromEdgeData = d_pointMappedFromEdge;

    unsigned int last = pointSourceData.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputPoints( index[i], last );
        int source = pointSourceData[last];
//			sout << "remove output point " << last << " from source " << source << sendl;
        if (source > 0)
        {
            pointMappedFromPointData[source-1] = -1;
        }
        else if (source < 0)
        {
            pointMappedFromEdgeData[-source-1] = -1;
        }
        --last;
    }

    pointSourceData.resize( pointSourceData.size() - index.size() );

}

void SimpleTesselatedTetraTopologicalMapping::renumberOutputPoints( const sofa::helper::vector<unsigned int>& index )
{
    helper::ReadAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        setPointSource(i, pointSourceData[ index[i] ]);
    }
}


void SimpleTesselatedTetraTopologicalMapping::swapOutputTetrahedra(int i1, int i2)
{
    // first update (*pointSourceData)

    helper::vector<int>& tetraSourceData = *(tetraSource.beginEdit());

    int i1Source = tetraSourceData[i1];
    int i2Source = tetraSourceData[i2];
    tetraSourceData[i1] = i2Source;
    tetraSourceData[i2] = i1Source;

    helper::vector< fixed_array<int, 8> >& tetrahedraMappedFromTetraData = *(tetrahedraMappedFromTetra.beginEdit());

    if (i1Source != -1)
        for (int j=0; j<8; ++j)
            if (tetrahedraMappedFromTetraData[i1Source][j] == i1)
            {
                tetrahedraMappedFromTetraData[i1Source][j] = i2;
                break;
            }
    if (i2Source != -1)
        for (int j=0; j<8; ++j)
            if (tetrahedraMappedFromTetraData[i2Source][j] == i2)
            {
                tetrahedraMappedFromTetraData[i2Source][j] = i1;
                break;
            }
    tetrahedraMappedFromTetra.endEdit();
    tetraSource.endEdit();
}

void SimpleTesselatedTetraTopologicalMapping::removeOutputTetrahedra( const sofa::helper::vector<unsigned int>& index )
{

    helper::vector< fixed_array<int, 8> >& tetrahedraMappedFromTetraData = *(tetrahedraMappedFromTetra.beginEdit());
    helper::vector<int>& tetraSourceData = *(tetraSource.beginEdit());

    int last = tetraSourceData.size() -1;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapOutputTetrahedra( index[i], last );
        int source = tetraSourceData[last];
        if (source != -1)
        {
			int nbt = 0;

            for (int j=0; j<8; ++j)
			{
                if (tetrahedraMappedFromTetraData[source][j] == last)
                {
                    tetrahedraMappedFromTetraData[source][j] = -1;
                }
                else if (tetrahedraMappedFromTetraData[source][j] != -1)
                    ++nbt;
			}
            if (nbt == 0) // we need to remove the source tetra
            {
//				sout << "SimpleTesselatedTetraTopologicalMapping: source tetra "<<source<<" needs to be removed."<<sendl;
                tetrahedraToRemove.insert(source);
            }
            else
            {
//			    sout << "SimpleTesselatedTetraTopologicalMapping: source tetra "<<source<<" now has "<<nbt<<" / 8 childs."<<sendl;
            }
            --last;
        }
    }

    tetraSourceData.resize( tetraSourceData.size() - index.size() );

    tetrahedraMappedFromTetra.endEdit();
    tetraSource.endEdit();
}



void SimpleTesselatedTetraTopologicalMapping::updateTopologicalMappingTopDown()
{
    if(fromModel && toModel)
    {
        //TetrahedronSetTopologyContainer *from_tstc;
        //fromModel->getContext()->get(from_tstc);
        //TriangleSetTopologyModifier *from_tstm;
        //fromModel->getContext()->get(from_tstm);
        //TetrahedronSetTopologyContainer *to_tstc;
        //toModel->getContext()->get(to_tstc);

        std::list<const TopologyChange *>::const_iterator changeIt=fromModel->beginChange();
        std::list<const TopologyChange *>::const_iterator itEnd=fromModel->endChange();

        while( changeIt != itEnd )
        {
            TopologyChangeType changeType = (*changeIt)->getChangeType();

            switch( changeType )
            {
            case core::topology::POINTSINDICESSWAP:
            {
                unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
                unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];
                // i1 and i2 swapped in input model
//				 sout << "INPUT SWAP POINTS "<<i1 << " " << i2 << sendl;
                swapInputPoints(i1,i2);
                break;
            }
            case core::topology::POINTSADDED:
            {
                /// @todo
                break;
            }
            case core::topology::POINTSREMOVED:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();
//				 sout << "INPUT REMOVE POINTS "<<tab << sendl;
                removeInputPoints( tab );
                break;
            }
            case core::topology::POINTSRENUMBERING:
            {
                const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getinv_IndexArray();
//				 sout << "INPUT RENUMBER POINTS "<<tab << sendl;
                renumberInputPoints( tab );
                break;
            }
            case core::topology::EDGESADDED:
            {
                /// @todo
                break;
            }
            case core::topology::EDGESREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE EDGES "<<tab << sendl;
                removeInputEdges( tab );
                break;
            }
            case core::topology::TETRAHEDRAADDED:
            {
                /// @todo
                break;
            }
            case core::topology::TETRAHEDRAREMOVED:
            {
                const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();
//				sout << "INPUT REMOVE TETRAHEDRA "<<tab << sendl;
                removeInputTetrahedra( tab );
                break;
            }
            case core::topology::ENDING_EVENT:
            {
                //sout << "(*pointMappedFromPointData) = " << (*pointMappedFromPointData)<<sendl;
                //sout << "pointMappedFromEdge = " << pointMappedFromEdge<<sendl;
                //sout << "(*pointSourceData) = " << pointMappedFromEdge<<sendl;
                break;
            }
            default: break;

            }
            ++changeIt;
        }
    }
}

void SimpleTesselatedTetraTopologicalMapping::swapInputPoints(int i1, int i2)
{
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;

    int i1Map = pointMappedFromPointData[i1];
    int i2Map = pointMappedFromPointData[i2];
    pointMappedFromPointData[i1] = i2Map;
    if (i2Map != -1) pointSourceData[i2Map] = i1+1;
    pointMappedFromPointData[i2] = i1Map;
    if (i1Map != -1) pointSourceData[i1Map] = i2+1;

}

void SimpleTesselatedTetraTopologicalMapping::removeInputPoints( const sofa::helper::vector<unsigned int>& index )
{
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;

    unsigned int last = pointMappedFromPointData.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputPoints( index[i], last );
        int map = pointMappedFromPointData[last];
        if (map != -1)
            pointSourceData[map] = 0;
        --last;
    }

    pointMappedFromPointData.resize( last + 1 );
}

void SimpleTesselatedTetraTopologicalMapping::renumberInputPoints( const sofa::helper::vector<unsigned int>& index )
{
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromPointData = d_pointMappedFromPoint;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        int map = pointMappedFromPointData[index[i]];
        pointMappedFromPointData[i] = map;
        if (map != -1)
            pointSourceData[map] = i+1;
    }

}

void SimpleTesselatedTetraTopologicalMapping::swapInputEdges(int i1, int i2)
{
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromEdgeData = d_pointMappedFromEdge;

    int i1Map = pointMappedFromEdgeData[i1];
    int i2Map = pointMappedFromEdgeData[i2];
    pointMappedFromEdgeData[i1] = i2Map;
    if (i2Map != -1) pointSourceData[i2Map] = -1-i1;
    pointMappedFromEdgeData[i2] = i1Map;
    if (i1Map != -1) pointSourceData[i1Map] = -1-i2;

}

void SimpleTesselatedTetraTopologicalMapping::removeInputEdges( const sofa::helper::vector<unsigned int>& index )
{

    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointSourceData = d_pointSource;
    helper::WriteAccessor< Data< sofa::helper::vector<int> > > pointMappedFromEdgeData = d_pointMappedFromEdge;

    unsigned int last = pointMappedFromEdgeData.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputEdges( index[i], last );
        int map = pointMappedFromEdgeData[last];
        if (map != -1)
            pointSourceData[map] = 0;
        --last;
    }

    pointMappedFromEdgeData.resize( last + 1 );

}


void SimpleTesselatedTetraTopologicalMapping::swapInputTetrahedra(int i1, int i2)
{
    helper::vector< fixed_array<int, 8> >& tetrahedraMappedFromTetraData = *(tetrahedraMappedFromTetra.beginEdit());
    helper::vector<int>& tetraSourceData = *(tetraSource.beginEdit());

    fixed_array<int, 8> i1Map = tetrahedraMappedFromTetraData[i1];
    fixed_array<int, 8> i2Map = tetrahedraMappedFromTetraData[i2];
    tetrahedraMappedFromTetraData[i1] = i2Map;
    for (int j=0; j<8; ++j)
        if (i2Map[j] != -1) tetraSourceData[i2Map[j]] = i1;
    tetrahedraMappedFromTetraData[i2] = i1Map;
    for (int j=0; j<8; ++j)
        if (i1Map[j] != -1) tetraSourceData[i1Map[j]] = i2;

    tetrahedraMappedFromTetra.endEdit();
    tetraSource.endEdit();
}


void SimpleTesselatedTetraTopologicalMapping::removeInputTetrahedra( const sofa::helper::vector<unsigned int>& index )
{
    helper::vector< fixed_array<int, 8> >& tetrahedraMappedFromTetraData = *(tetrahedraMappedFromTetra.beginEdit());
    helper::vector<int>& tetraSourceData = *(tetraSource.beginEdit());

    unsigned int last = tetrahedraMappedFromTetraData.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        swapInputTetrahedra( index[i], last );
        fixed_array<int, 8> map = tetrahedraMappedFromTetraData[last];
        for (int j=0; j<8; ++j)
            if (map[j] != -1)
                tetraSourceData[map[j]] = -1;
        --last;
    }

    tetrahedraMappedFromTetraData.resize( last + 1 );

    tetrahedraMappedFromTetra.endEdit();
    tetraSource.endEdit();
}



} // namespace topology
} // namespace component
} // namespace sofa

