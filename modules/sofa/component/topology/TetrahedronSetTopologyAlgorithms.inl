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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/TetrahedronSetTopologyAlgorithms.h>
#include <algorithm>
#include <functional>
#include <sofa/component/topology/TetrahedronSetTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template<class DataTypes>
TetrahedronSetTopology< DataTypes >* TetrahedronSetTopologyAlgorithms< DataTypes >::getTetrahedronSetTopology() const
{
    return static_cast<TetrahedronSetTopology< DataTypes >* > (this->m_basicTopology);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeTetrahedra(sofa::helper::vector< unsigned int >& tetrahedra)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyModifier* modifier  = topology->getTetrahedronSetTopologyModifier();
    TetrahedronSetTopologyContainer* container = topology->getTetrahedronSetTopologyContainer();

    modifier->removeTetrahedraWarning(tetrahedra);

    // inform other objects that the triangles are going to be removed
    container->propagateTopologicalChanges();

    // now destroy the old tetrahedra.
    modifier->removeTetrahedraProcess(  tetrahedra ,true);

    container->checkTopology();
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeTetrahedra(items);
}

template<class DataTypes>
void TetrahedronSetTopologyAlgorithms< DataTypes >::RemoveTetraBall(unsigned int ind_ta, unsigned int ind_tb)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();

    sofa::helper::vector<unsigned int> init_indices;
    sofa::helper::vector<unsigned int> &indices = init_indices;
    topology->getTetrahedronSetGeometryAlgorithms()->getTetraInBall(ind_ta, ind_tb, indices);
    removeTetrahedra(indices);

    //cout<<"INFO, number to remove = "<< indices.size() <<endl;
}

template<class DataTypes>
void  TetrahedronSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    TetrahedronSetTopology< DataTypes > *topology = getTetrahedronSetTopology();
    TetrahedronSetTopologyModifier* modifier  = topology->getTetrahedronSetTopologyModifier();
    TetrahedronSetTopologyContainer* container = topology->getTetrahedronSetTopologyContainer();

    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    container->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    container->checkTopology();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TETEAHEDRONSETTOPOLOGYALGORITHMS_INL
