/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
    , valuesIndexedInRegularGrid( initData ( &valuesIndexedInRegularGrid, sofa::helper::vector<unsigned char>(), "valuesIndexedInRegularGrid", "values indexed in the Regular Grid" ) )
    , valuesIndexedInTopology( initData(&valuesIndexedInTopology, "valuesIndexedInTopology", "values indexed in the topology"))
    , idxInRegularGrid( initData ( &idxInRegularGrid, sofa::helper::vector<BaseMeshTopology::HexaID>(), "idxInRegularGrid", "indices in the Regular Grid" ) )
    , idInRegularGrid2IndexInTopo( initData ( &idInRegularGrid2IndexInTopo, std::map< unsigned int, BaseMeshTopology::HexaID> (), "idInRegularGrid2IndexInTopo", "map between id in the Regular Grid and index in the topology" ) )
    , voxelSize( initData(&voxelSize, defaulttype::Vector3(1,1,1), "voxelSize", "Size of the Voxels"))
{
    valuesIndexedInRegularGrid.setDisplayed( false);
    valuesIndexedInTopology.setDisplayed( false);
    idInRegularGrid2IndexInTopo.setDisplayed( false);
}

void DynamicSparseGridTopologyContainer::loadFromMeshLoader ( sofa::component::container::MeshLoader* loader )
{

    if (!valuesIndexedInRegularGrid.getValue().empty()) return;
    HexahedronSetTopologyContainer::loadFromMeshLoader( loader);
    sofa::component::container::VoxelGridLoader* voxelGridLoader = dynamic_cast< sofa::component::container::VoxelGridLoader*> ( loader );
    if ( !voxelGridLoader )
    {
        this->serr << "DynamicSparseGridTopologyContainer::loadFromMeshLoader(): The loader used is not a VoxelGridLoader ! You must use it for this topology." << this->sendl;
        exit(0);
    }

    // Init regular/topo mapping
    helper::vector<BaseMeshTopology::HexaID>& iirg = *idxInRegularGrid.beginEdit();
    std::map< unsigned int, BaseMeshTopology::HexaID> &idrg2tpo = *idInRegularGrid2IndexInTopo.beginEdit();
    helper::vector<unsigned char>& viirg = *(valuesIndexedInRegularGrid.beginEdit());
    helper::vector<unsigned char>& viit = *(valuesIndexedInTopology.beginEdit());

    voxelGridLoader->getIndicesInRegularGrid( iirg);
    for( unsigned int i = 0; i < iirg.size(); i++)
    {
        idrg2tpo.insert( make_pair( iirg[i], i ));
    }

    // Init values
    int dataSize = voxelGridLoader->getDataSize();
    unsigned char* data = voxelGridLoader->getData();

    // init values in regular grid. (dense).
    viirg.resize( dataSize);
    //for( int i = 0; i < dataSize; i++)
    //  viirg[i] = data[i];
    for( unsigned int i = 0; i < iirg.size(); ++i)
    {
        viirg[iirg[i]] = 255;
    }

    // init values in topo. (pas dense).
    viit.resize( iirg.size());
    for(unsigned int i = 0; i < iirg.size(); i++)
    {
        viit[i] = data[iirg[i]];
    }

    // init resolution & voxelSize.
    Vec3i& res = *resolution.beginEdit();
    voxelGridLoader->getResolution ( res );
    resolution.endEdit();
    idxInRegularGrid.endEdit();
    idInRegularGrid2IndexInTopo.endEdit();
    valuesIndexedInRegularGrid.endEdit();
    valuesIndexedInTopology.endEdit();
    defaulttype::Vector3 &value=*voxelSize.beginEdit();
    voxelGridLoader->getVoxelSize ( value );
    voxelSize.endEdit();
}

} // namespace topology

} // namespace component

} // namespace sofa

