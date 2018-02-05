/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_QUADBENDINGSPRINGS_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_QUADBENDINGSPRINGS_INL

#include <SofaGeneralDeformable/QuadBendingSprings.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
QuadBendingSprings<DataTypes>::QuadBendingSprings()
    : sofa::component::interactionforcefield::StiffSpringForceField<DataTypes>()
    , localRange( initData(&localRange, defaulttype::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
{
}


template<class DataTypes>
QuadBendingSprings<DataTypes>::~QuadBendingSprings()
{}

template<class DataTypes>
void QuadBendingSprings<DataTypes>::addSpring( unsigned a, unsigned b, std::set<IndexPair>& springSet )
{
    if (localRange.getValue()[0] >= 0)
    {
        if ((int)a < localRange.getValue()[0] && (int)b < localRange.getValue()[0]) return;
    }
    if (localRange.getValue()[1] >= 0)
    {
        if ((int)a > localRange.getValue()[1] && (int)b > localRange.getValue()[1]) return;
    }
    IndexPair ab(a<b?a:b, a<b?b:a);
    if (springSet.find(ab) != springSet.end()) return;
    springSet.insert(ab);
    const VecCoord& x =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    Real s = (Real)this->ks.getValue();
    Real d = (Real)this->kd.getValue();
    Real l = (x[a]-x[b]).norm();
    this->SpringForceField<DataTypes>::addSpring(a,b, s, d, l );
    //sout<<"=================================QuadBendingSprings<DataTypes>::addSpring "<<a<<", "<<b<<sendl;
}

template<class DataTypes>
void QuadBendingSprings<DataTypes>::registerEdge( IndexPair ab, IndexPair cd, std::map<IndexPair, IndexPair>& edgeMap, std::set<IndexPair>& springSet)
{
    if (ab.first > ab.second)
    {
        ab = std::make_pair(ab.second,ab.first);
        cd = std::make_pair(cd.second,cd.first);
    }
    if (edgeMap.find(ab) == edgeMap.end())
    {
        edgeMap[ab] = cd;
    }
    else
    {
        // create a spring between the opposite
        IndexPair ef = edgeMap[ab];
        this->addSpring(cd.first, ef.first, springSet);
        this->addSpring(cd.second, ef.second, springSet);
    }
}

template<class DataTypes>
void QuadBendingSprings<DataTypes>::init()
{
    // Set the bending springs

    std::map< IndexPair, IndexPair > edgeMap;
    std::set< IndexPair > springSet;

    sofa::core::topology::BaseMeshTopology* topology = this->getContext()->getMeshTopology();
    assert( topology );

    const sofa::core::topology::BaseMeshTopology::SeqQuads& quads = topology->getQuads();
    //sout<<"==================================QuadBendingSprings<DataTypes>::init(), quads size = "<<quads.size()<<sendl;
    for( unsigned i= 0; i<quads.size(); ++i )
    {
        const sofa::core::topology::BaseMeshTopology::Quad& face = quads[i];
        {
            registerEdge( std::make_pair(face[0], face[1]), std::make_pair(face[3], face[2]), edgeMap, springSet );
            registerEdge( std::make_pair(face[1], face[2]), std::make_pair(face[0], face[3]), edgeMap, springSet );
            registerEdge( std::make_pair(face[2], face[3]), std::make_pair(face[1], face[0]), edgeMap, springSet );
            registerEdge( std::make_pair(face[3], face[0]), std::make_pair(face[2], face[1]), edgeMap, springSet );
        }
    }

    // init the parent class
    StiffSpringForceField<DataTypes>::init();
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_QUADBENDINGSPRINGS_INL */
