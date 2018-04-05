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

#include <SofaNonUniformFem/DynamicSparseGridTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/loader/VoxelLoader.h>
#include <SofaBaseTopology/TopologyData.inl>

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

void DynamicSparseGridTopologyContainer::init()
{
    HexahedronSetTopologyContainer::init();
    // Init regular/topo mapping
    sofa::core::loader::VoxelLoader* VoxelLoader;
    this->getContext()->get(VoxelLoader);
    if ( !VoxelLoader )
    {
        this->serr << "DynamicSparseGridTopologyContainer::init(): No VoxelLoader found! Aborting..." << this->sendl;
        exit(EXIT_FAILURE);
    }

    const helper::vector<BaseMeshTopology::HexaID>& iirg = idxInRegularGrid.getValue();
    std::map< unsigned int, BaseMeshTopology::HexaID> &idrg2tpo = *idInRegularGrid2IndexInTopo.beginEdit();
    helper::vector<unsigned char>& viirg = *(valuesIndexedInRegularGrid.beginEdit());
    helper::vector<unsigned char>& viit = *(valuesIndexedInTopology.beginEdit());

    for( unsigned int i = 0; i < iirg.size(); i++)
    {
        idrg2tpo.insert( make_pair( iirg[i], i ));
    }

    // Init values
    int dataSize = VoxelLoader->getDataSize();
    const unsigned char* data = VoxelLoader->getData();

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


    idInRegularGrid2IndexInTopo.endEdit();
    valuesIndexedInRegularGrid.endEdit();
    valuesIndexedInTopology.endEdit();

}



} // namespace topology

} // namespace component

} // namespace sofa

