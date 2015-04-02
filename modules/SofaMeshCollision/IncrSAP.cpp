/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <SofaMeshCollision/IncrSAP.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

double ISAPBox::tolerance = (double)(1e-7);

double ISAPBox::squaredDistance(const ISAPBox & other) const{
//    assert(dynamic_cast<OBBModel*>(cube.getCollisionModel()->getNext()) != 0x0);
//    assert(dynamic_cast<OBBModel*>(other.cube.getCollisionModel()->getNext()) != 0x0);
    const defaulttype::Vector3 & min_vect0 = cube.minVect();
    const defaulttype::Vector3 & max_vect0 = cube.maxVect();
    const defaulttype::Vector3 & min_vect1 = other.cube.minVect();
    const defaulttype::Vector3 & max_vect1 = other.cube.maxVect();

    double temp;
    double dist2 = 0;

    for(int i = 0 ; i < 3 ; ++i){
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        if(max_vect0[i] <= min_vect1[i]){
            temp = max_vect0[i] - min_vect1[i];
            dist2 += temp * temp;
        }
        else if(max_vect1[i] <= min_vect0[i]){
            temp = max_vect1[i] - min_vect0[i];
            dist2 += temp * temp;
        }
    }

    return dist2;
}

bool ISAPBox::overlaps(const ISAPBox & other,double alarmDist) const{
//    assert(dynamic_cast<OBBModel*>(cube.getCollisionModel()->getNext()) != 0x0);
//    assert(dynamic_cast<OBBModel*>(other.cube.getCollisionModel()->getNext()) != 0x0);
    const defaulttype::Vector3 & min_vect0 = cube.minVect();
    const defaulttype::Vector3 & max_vect0 = cube.maxVect();
    const defaulttype::Vector3 & min_vect1 = other.cube.minVect();
    const defaulttype::Vector3 & max_vect1 = other.cube.maxVect();

    for(int i = 0 ; i < 3 ; ++i){
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        if(max_vect0[i] + alarmDist <= min_vect1[i] || max_vect1[i] + alarmDist <= min_vect0[i])
            return false;
    }

    return true;
}


using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace collision;

SOFA_DECL_CLASS(TIncrSAP)

int IncrSAPClassSofaVector = core::RegisterObject("Collision detection using incremental sweep and prune")
        .addAlias( "IncrSAP" )
        .addAlias( "IncrementalSAP" )
        .addAlias( "IncrementalSweepAndPrune" )
        .add< IncrSAP >( true )
        .add< TIncrSAP<helper::vector,helper::CPUMemoryManager> >()
        ;


template class SOFA_MESH_COLLISION_API TIncrSAP<helper::vector,helper::CPUMemoryManager>;
template class SOFA_MESH_COLLISION_API TIncrSAP<std::vector,std::allocator>;

} // namespace collision

} // namespace component

} // namespace sofa

