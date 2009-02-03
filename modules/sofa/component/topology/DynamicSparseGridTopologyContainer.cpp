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

#include <sofa/component/topology/DynamicSparseGridTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/VoxelGridLoader.h>

namespace sofa
{
namespace component
{
namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS ( DynamicSparseGridTopologyContainer );
int DynamicSparseGridTopologyContainerClass = core::RegisterObject ( "Hexahedron set topology container" )
        .add< DynamicSparseGridTopologyContainer >()
        ;

DynamicSparseGridTopologyContainer::DynamicSparseGridTopologyContainer()
    : HexahedronSetTopologyContainer()
    , resolution ( initData ( &resolution, Vec3i ( 0, 0, 0 ), "resolution", "voxel grid resolution" ) )
{
}

DynamicSparseGridTopologyContainer::DynamicSparseGridTopologyContainer ( const sofa::helper::vector< Hexahedron > &hexahedra )
    : HexahedronSetTopologyContainer ( hexahedra )
    , resolution ( initData ( &resolution, Vec3i ( 0, 0, 0 ), "resolution", "voxel grid resolution" ) )
{
}

void DynamicSparseGridTopologyContainer::loadFromMeshLoader ( sofa::component::MeshLoader* loader )
{
    sofa::component::VoxelGridLoader* voxelGridLoader = dynamic_cast< sofa::component::VoxelGridLoader*> ( loader );
    if ( !voxelGridLoader )
    {
        cerr << "DynamicSparseGridTopologyContainer::loadFromMeshLoader(): The loader used is not a VoxelGridLoader ! You must use it for this topology." << endl;
        return;
    }

    // load points
    PointSetTopologyContainer::loadFromMeshLoader ( voxelGridLoader );
    d_hexahedron.beginEdit();

    helper::vector<BaseMeshTopology::HexaID>& iirg = *(idxInRegularGrid.beginEdit());

    voxelGridLoader->getIndicesInRegularGrid( iirg);
    for( unsigned int i = 0; i < iirg.size(); i++)
    {
        idInRegularGrid2Hexa.insert( make_pair( iirg[i], i ));
    }

    idxInRegularGrid.endEdit();

    int dataSize = voxelGridLoader->getDataSize();
    unsigned char* data = voxelGridLoader->getData();
    valuesIndexedInRegularGrid.resize( dataSize);

    helper::vector<unsigned char>& viit = *(valuesIndexedInTopology.beginEdit());

    viit.resize( dataSize);
    for( int i = 0; i < dataSize; i++)
    {
        valuesIndexedInRegularGrid[i] = data[i];
        viit[i] = data[i];
    }

    valuesIndexedInTopology.endEdit();

    voxelGridLoader->getHexas( m_hexahedron);
    d_hexahedron.endEdit();

    Vec3i& res = *resolution.beginEdit();
    voxelGridLoader->getResolution ( res );
    resolution.endEdit();

    voxelGridLoader->getVoxelSize ( voxelSize );
}

} // namespace topology

} // namespace component

} // namespace sofa

