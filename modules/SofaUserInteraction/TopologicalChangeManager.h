/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_TOPOLOGICALCHANGEMANAGER_H
#define SOFA_COMPONENT_COLLISION_TOPOLOGICALCHANGEMANAGER_H
#include "config.h"

#include <sofa/core/CollisionElement.h>

#include <sofa/core/BehaviorModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaUserInteraction/CuttingPoint.h>


#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/Node.h>


namespace sofa
{

namespace component
{

namespace collision
{

#if 0
class TetrahedronModel;
#endif


/// a class to manage the handling of topological changes which have been requested from the Collision Model
class SOFA_USER_INTERACTION_API TopologicalChangeManager
{
public:
    TopologicalChangeManager();
    ~TopologicalChangeManager();

    /// Handles Removing of topological element (from any type of topology)
    int removeItemsFromCollisionModel(sofa::core::CollisionElementIterator) const;
    int removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const int& index) const;
    int removeItemsFromCollisionModel(sofa::core::CollisionModel* model, const helper::vector<int>& indices) const;


    /** Handles Cutting (activated only for a triangular topology)
     *
     * Only one model is given. This function perform incision beetween input point and stocked
     * informations. If it is the first point of the incision, these informations are stocked.
     * i.e element index and picked point coordinates.
     *
     * \sa incisionTriangleSetTopology
     *
     * @param elem - iterator to collision model.
     * @param pos - picked point coordinates.
     * @param firstInput - bool, if yes this is the first incision point.
     * @param snapingValue - threshold distance from point to incision path where point has to be snap on incision path.
     * @param snapingBorderValue - threshold distance from point to mesh border where incision is considered to reach the border..
     *
     * @return bool - true if incision has been performed.
     */
    bool incisionCollisionModel(sofa::core::CollisionElementIterator elem,
                                defaulttype::Vector3& pos, bool firstInput,
                                int snapingValue = 0,
                                int snapingBorderValue = 0);


    /** Handles Cutting for general model collision (activated only for a triangular topology for the moment).
     *
     * Given two collision model, perform an incision between two points.
     * \sa incisionTriangleSetTopology
     *
     * @param model1 - first collision model.
     * @param idx1 - first element index.
     * @param firstPoint - first picked point coordinates.
     * @param model2 - second collision model.
     * @param idx2 - second element index.
     * @param secondPoint - second picked point coordinates.
     * @param snapingValue - threshold distance from point to incision path where point has to be snap on incision path.
     * @param snapingBorderValue - threshold distance from point to mesh border where incision is considered to reach the border..
     *
     * @return bool - true if incision has been performed.
     */
    bool incisionCollisionModel(sofa::core::CollisionModel* model1,
                                unsigned int idx1,
                                const defaulttype::Vector3& firstPoint,
                                sofa::core::CollisionModel *model2,
                                unsigned int idx2,
                                const defaulttype::Vector3& secondPoint,
                                int snapingValue = 0,
                                int snapingBorderValue = 0);

    /** Sets incision starting parameter - incision is just started or already in course
     *
     * @param isFirstCut - true if the next incision event will be the first of a new incision
     */
    void setIncisionFirstCut(bool isFirstCut);

protected:

private:

    /** Perform incision from point to point in a triangular mesh.
     *
     * \sa incisionCollisionModel
     *
     * @param model1 - first triangle collision model.
     * @param idx1 - first triangle index.
     * @param firstPoint - first picked point coordinates.
     * @param model2 - second triangle collision model.
     * @param idx2 - second triangle index.
     * @param secondPoint - second picked point coordinates.
     * @param snapingValue - threshold distance from point to incision path where point has to be snap on incision path.
     * @param snapingBorderValue - threshold distance from point to mesh border where incision is considered to reach the border..
     *
     * @return bool - true if incision has been performed.
     */
    bool incisionTriangleModel(TriangleModel* model1,
                               unsigned int idx1,
                               const defaulttype::Vector3& firstPoint,
                               TriangleModel *model2,
                               unsigned int idx2,
                               const defaulttype::Vector3& secondPoint,
                               int snapingValue = 0,
                               int snapingBorderValue = 0);



    int removeItemsFromTriangleModel(sofa::component::collision::TriangleModel* model, const helper::vector<int>& indices) const;

#if 0
    int removeItemsFromTetrahedronModel(sofa::component::collision::TetrahedronModel* model, const helper::vector<int>& indices) const;
#endif
    int removeItemsFromSphereModel(sofa::component::collision::SphereModel* model, const helper::vector<int>& indices) const;

private:
    /// Global variables to register intermediate informations for point to point incision.(incision along one segment in a triangular mesh)
    struct Incision
    {
        /// Temporary point index for successive incisions
        sofa::core::topology::BaseMeshTopology::PointID indexPoint;

        /// Temporary point coordinate for successive incisions
        defaulttype::Vector3 coordPoint;

        /// Temporary triangle index for successive incisions
        sofa::core::topology::BaseMeshTopology::TriangleID indexTriangle;

        /// Information of first incision for successive incisions
        bool firstCut;


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
