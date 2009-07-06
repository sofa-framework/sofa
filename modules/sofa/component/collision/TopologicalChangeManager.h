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

#ifndef SOFA_COMPONENT_COLLISION_TOPOLOGICALCHANGEMANAGER_H
#define SOFA_COMPONENT_COLLISION_TOPOLOGICALCHANGEMANAGER_H

#include <sofa/core/CollisionElement.h>

#include <sofa/core/BehaviorModel.h>

#ifdef SOFA_DEV
#include <sofa/component/collision/CuttingManager.h>
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>

/** a class to manage the handling of topological changes which have been requested from the Collision Model */

namespace sofa
{

namespace component
{
namespace collision
{
class TriangleModel;
class SphereModel;
}
}

namespace component
{

namespace collision
{
using namespace sofa::defaulttype;

class TopologicalChangeManager
{
public:
    TopologicalChangeManager();
    ~TopologicalChangeManager();

    /// Handles Removing of topological element (from any type of topology)
    void removeItemsFromCollisionModel(sofa::core::CollisionElementIterator) const;
    void removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const std::vector<int>& indices) const;

    /// Handles Cutting (activated only for a triangular topology), using global variables to register the two last input points
    bool incisionCollisionModel(sofa::core::CollisionElementIterator, Vector3&, bool, bool);

protected:

private:
    bool incisionTriangleModel(sofa::core::CollisionElementIterator, Vector3&, bool, bool);
    /// Intermediate method to handle cutting
    bool incisionTriangleSetTopology(sofa::core::componentmodel::topology::BaseMeshTopology*);
    bool incisionTriangleSetTopology(sofa::core::CollisionElementIterator, Vector3&, bool, bool, sofa::core::componentmodel::topology::BaseMeshTopology*);

    void removeItemsFromTriangleModel(sofa::component::collision::TriangleModel* model, const std::vector<int>& indices) const;
    void removeItemsFromSphereModel(sofa::component::collision::SphereModel* model, const std::vector<int>& indices) const;

private:
    /// Global variables to register the two last input points (for incision along one segment in a triangular mesh)
    struct Incision
    {
        Vector3 a_init;
        Vector3 b_init;
        unsigned int ind_ta_init;
        unsigned int ind_tb_init;

        bool is_first_cut;
        bool is_cut_completed;

        unsigned int b_last_init;
        sofa::helper::vector< unsigned int > b_p12_last_init;
        sofa::helper::vector< unsigned int > b_i123_last_init;

        unsigned int a_last_init;
        sofa::helper::vector< unsigned int >  a_p12_last_init;
        sofa::helper::vector< unsigned int >  a_i123_last_init;

#ifdef SOFA_DEV
        CuttingPoint* cutB;
        CuttingPoint* cutA;
#endif
    }	incision;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
