/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <CollisionOBBCapsule/detection/intersection/OBBIntersection.h>

#include <sofa/core/collision/IntersectorFactory.h>
#include <sofa/core/collision/Intersection.inl>

#include <sofa/component/collision/geometry/SphereModel.h>
#include <CollisionOBBCapsule/geometry/OBBModel.h>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>
#include <sofa/component/collision/geometry/RayModel.h>

#include <sofa/gui/component/performer/FixParticlePerformer.h>

namespace collisionobbcapsule::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;
using namespace collisionobbcapsule::geometry;

IntersectorCreator<DiscreteIntersection, RigidDiscreteIntersection> RigidDiscreteIntersectors("Rigid");
IntersectorCreator<NewProximityIntersection, RigidMeshDiscreteIntersection> RigidMeshDiscreteIntersectors("RigidMesh");

RigidDiscreteIntersection::RigidDiscreteIntersection(DiscreteIntersection* intersection)
{
    intersection->intersectors.add<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, RigidDiscreteIntersection>(this);
    intersection->intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, RigidDiscreteIntersection>(this);
    intersection->intersectors.add<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, RigidDiscreteIntersection>(this);

    intersection->intersectors.add<RayCollisionModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, RigidDiscreteIntersection>(this);
}

RigidMeshDiscreteIntersection::RigidMeshDiscreteIntersection(NewProximityIntersection* intersection)
{
    intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, RigidMeshDiscreteIntersection>(this);
}

bool RigidDiscreteIntersection::testIntersection(Ray& /*rRay*/, OBB& /*rOBB*/, const core::collision::Intersection*)
{
    return false;
}

static float const c_fMin = -3.402823466e+38f;
static float const c_fMax = 3.402823466e+38f;
typedef Mat<3, 3, SReal> Mat33;

int  RigidDiscreteIntersection::computeIntersection(Ray& rRay, OBB& rObb, OutputVector* contacts, const core::collision::Intersection*)
{
    //Near intersection: ray to closest plat of slab
    float fNear = c_fMin;

    //Far intersection: ray to farthest plane of slab
    float fFar = c_fMax;

    //Epsilon: for checks between planes and direction of ray
    float fEPSILON = 1e-4f;

    //Ray
    const type::Vec3& v3Origin = rRay.origin();
    const type::Vec3& v3Direction = rRay.direction();

    //Box
    const type::Vec3 v3HalfExtents = rObb.extents();
    const type::Vec3& v3BoxCenter = rObb.center();
    const Quat<SReal>& qOrientation = rObb.orientation();
    Mat33 m33Orientation;
    qOrientation.toMatrix(m33Orientation);

    //Vector from origin of ray to center of box
    type::Vec3 v3RayOriginToBoxCenter = v3BoxCenter - v3Origin;

    //Normal at near and far intersection points
    type::Vec3 v3NormalAtNear(0.0f, 1.0f, 0.0f);
    type::Vec3 v3NormalAtFar(0.0f, 1.0f, 0.0f);

    //For the 3 slabs
    for (unsigned int i = 0; i < 3; i++)
    {
        type::Vec3 v3CurrAxis = m33Orientation.col(i); //TODO: implement the return of a reference instead of a copy of the column
        float fR = (float)(v3CurrAxis * v3RayOriginToBoxCenter);
        float fBoxMax = (float)v3HalfExtents[i];
        float fBoxMin = (float)-v3HalfExtents[i];
        float fNormal = 1.f;

        //Check if planes are parallel
        if (fabs(v3Direction * v3CurrAxis) < fEPSILON)
        {
            if (fBoxMin > fR || fBoxMax < fR)
            {
                return 0;
            }
        }
        else
        {
            // Ray not parallel to planes, so find intersection parameters
            float fS = (float)(v3Direction * v3CurrAxis);
            float fT0 = (fR + fBoxMax) / fS;
            float fT1 = (fR + fBoxMin) / fS;

            // Check ordering
            if (fT0 > fT1)
            {
                //Swap them
                float fTemp = fT0;
                fT0 = fT1;
                fT1 = fTemp;
                fNormal = -1.f;
            }

            //Compare with current values
            if (fT0 > fNear)
            {
                fNear = fT0;
                v3NormalAtNear = v3CurrAxis * fNormal;
            }
            if (fT1 < fFar)
            {
                fFar = fT1;
                v3NormalAtFar = v3CurrAxis * fNormal;
            }

            //Check if ray misses entirely, i.e. Exits with no collision as soon as slab intersection becomes empty
            if (fNear > fFar)
            {
                return 0;
            }

            // Ray starts after box, returns directly
            if (fFar < 0)
            {
                return 0;
            }

            // Ray ends before box, returns directly
            if (fNear > rRay.l())
            {
                return 0;
            }
        }
    }  //end of slabs checking

    bool bHit = false;

    float fHitFraction = 0;
    type::Vec3 v3Normal;
    type::Vec3 v3HitLocation;
    // If ray starts inside box
    if (fNear < 0.f)
    {
        // Make sure it does not ends strictly inside box
        if (fFar <= rRay.l())
        {
            bHit = true;
            fHitFraction = fFar / (float)rRay.l();
            v3Normal = v3NormalAtFar;
            v3HitLocation = rRay.origin() + rRay.direction() * rRay.l() * fHitFraction;
        }

    }
    else
    {
        // Filter hits if very close to each other
        if (fabs(fNear - fFar) < fEPSILON)
        {
            bHit = true;
            fHitFraction = fNear / (float)rRay.l();
            v3Normal = v3NormalAtNear;
            v3HitLocation = rRay.origin() + rRay.direction() * rRay.l();
        }
        else
        {
            bHit = true;
            fHitFraction = fNear / (float)rRay.l();
            v3Normal = v3NormalAtNear;
            v3HitLocation = rRay.origin() + rRay.direction() * fNear;

            // Ignore far hit if ends inside box
            //if(fFar <= rRay.l())
            //{
            //	bHit = true;
            //	fHitFraction = fFar / rRay.l();
            //	v3Normal = v3NormalAtFar;
            //	v3HitLocation = rRay.origin() + rRay.direction() * fFar;
            //}
        }
    }

    if (bHit)
    {
        //		const SReal contactDist = fHitFraction;
        contacts->resize(contacts->size() + 1);
        DetectionOutput* detection = &*(contacts->end() - 1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(rRay, rObb);
        detection->point[1] = v3HitLocation;
        detection->point[0] = v3HitLocation;
        detection->normal = v3Normal;
        detection->value = fHitFraction;
        detection->value -= fHitFraction;

        return 1;
    }


    return 0;

}

// add OBBModel to the list of supported collision models for FixParticlePerformer
using FixParticlePerformer3d = sofa::gui::component::performer::FixParticlePerformer<defaulttype::Vec3Types>;

int obbFixParticle = FixParticlePerformer3d::RegisterSupportedModel<OBBCollisionModel<sofa::defaulttype::Rigid3Types>>(
    []
(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, type::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        auto* obb = dynamic_cast<OBBCollisionModel<sofa::defaulttype::Rigid3Types>*>(model.get());

        if (!obb)
            return false;

        auto* collisionState= model->getContext()->getMechanicalState();
        fixPoint[0] = collisionState->getPX(idx);
        fixPoint[1] = collisionState->getPY(idx);
        fixPoint[2] = collisionState->getPZ(idx);

        points.push_back(idx);

        return true;
    }
);

} // namespace collisionobbcapsule::detection::intersection
