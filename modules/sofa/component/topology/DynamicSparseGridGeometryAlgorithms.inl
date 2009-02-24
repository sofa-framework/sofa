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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/DynamicSparseGridGeometryAlgorithms.h>
#include <sofa/component/topology/CommonAlgorithms.h>
#include <sofa/component/container/MechanicalObject.h>

namespace sofa
{

namespace component
{

namespace topology
{

template < class DataTypes >
void DynamicSparseGridGeometryAlgorithms<DataTypes>::init()
{
    HexahedronSetGeometryAlgorithms<DataTypes>::init();
    this->getContext()->get ( topoContainer );
    if ( !topoContainer )
    {
        cerr << "Hexa2TriangleTopologicalMapping::buildTriangleMesh(). Error: can't find the mapping on the triangular topology." << endl;
        exit(0);
    }
}

template < class DataTypes >
unsigned int DynamicSparseGridGeometryAlgorithms<DataTypes>::getTopoIndexFromRegularGridIndex ( unsigned int index )
{
    std::map< unsigned int, BaseMeshTopology::HexaID>::iterator it = topoContainer->idInRegularGrid2IndexInTopo.find( index);
    if( it == topoContainer->idInRegularGrid2IndexInTopo.end())
    {
        cerr << "DynamicSparseGridGeometryAlgorithms<DataTypes>::getTopoIndexFromRegularGridIndex(): Warning ! unexisting index given !" << endl;
    }
    return it->second;
}

template < class DataTypes >
int DynamicSparseGridGeometryAlgorithms<DataTypes>::findNearestElementInRestPos(const Coord& pos, Vector3& baryC, Real& distance) const
{
    int index = -1;
    distance = 1e10;

    Vec3i resolution = topoContainer->resolution.getValue();
    Vec3i tmp = Vec3i( (int)(pos[0] / topoContainer->voxelSize[0]), (int)(pos[1] / topoContainer->voxelSize[1]), (int)(pos[2] / topoContainer->voxelSize[2]));

    // Projection sur la bbox si l'element est en dehors.
    if( tmp[0] < 0) tmp[0] = 0;
    if( tmp[1] < 0) tmp[1] = 0;
    if( tmp[2] < 0) tmp[2] = 0;
    if( tmp[0] > resolution[0]) tmp[0] = resolution[0];
    if( tmp[1] > resolution[1]) tmp[1] = resolution[1];
    if( tmp[2] > resolution[2]) tmp[2] = resolution[2];

    const std::map< unsigned int, BaseMeshTopology::HexaID>& regular2topo = topoContainer->idInRegularGrid2IndexInTopo;
    unsigned int regularGridIndex;
    std::map< unsigned int, BaseMeshTopology::HexaID>::const_iterator it;
    for( int k = 0; k < 3; k++) // TODO on peut meme passer de 3 a 2 en enlevant les -1 dans les formules en dessous. A tester qd ca marchera.
    {
        if((((int)tmp[2])-1+k < 0) || (tmp[2]-1+k > resolution[2])) continue;
        for( int j = 0; j < 3; j++)
        {
            if((((int)tmp[1])-1+j < 0) || (tmp[1]-1+j > resolution[1])) continue;
            for( int i = 0; i < 3; i++)
            {
                if((((int)tmp[0])-1+i < 0) || (tmp[0]-1+i > resolution[0])) continue;
                regularGridIndex = (tmp[0]-1+i) + (tmp[1]-1+j)*resolution[0] + (tmp[2]-1+k)*resolution[0]*resolution[1];
                it = regular2topo.find( regularGridIndex);
                if( it != regular2topo.end())
                {
                    const Real d = computeElementRestDistanceMeasure(it->second, pos);
                    if(d<distance)
                    {
                        distance = d;
                        index = it->second;
                    }
                }
            }
        }
    }
    if( index == -1)
    {
        // Dans le cas de projection ou autre.... il se peut que la zone ciblée ne contienne pas d'hexas, il faut alors tous les parcourrir.
        std::cout << "DynamicSparseGridGeometryAlgorithms<DataTypes>::findNearestElementInRestPos(). Index non trouvé. Recherche dans tous les hexas ! (ceci nuit gravement aux perfs. vérifier les 'pos' données en entrée)." << std::endl;
        return HexahedronSetGeometryAlgorithms<DataTypes>::findNearestElementInRestPos( pos, baryC, distance);
    }

    distance = computeElementRestDistanceMeasure( index, pos);

    baryC = computeHexahedronRestBarycentricCoeficients(index, pos);

    return index;
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
