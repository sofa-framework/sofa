#include "CudaDistanceGridCollisionModel.h"
#include "CudaMechanicalObject.h"
#include <sofa/component/collision/NewProximityIntersection.inl>
#include <sofa/component/collision/DiscreteIntersection.inl>
#include <sofa/component/collision/RayPickInteractor.h>
#include <sofa/component/collision/RayContact.h>
#include "CudaContactMapper.h"
#include <sofa/component/collision/BarycentricPenalityContact.inl>
#include <fstream>
#include <GL/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

} //namespace collision


} //namespace component


namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaCollision)

using namespace sofa::component::collision;
typedef sofa::component::collision::TSphereModel<CudaVec3fTypes> CudaSphereModel;

class CudaProximityIntersection : public sofa::component::collision::NewProximityIntersection
{
public:

    virtual void init()
    {
        sofa::component::collision::NewProximityIntersection::init();
        intersectors.add<CudaSphereModel, RayModel,          DiscreteIntersection, true>(this);
        intersectors.add<CudaSphereModel, RayPickInteractor, DiscreteIntersection, true>(this);
        //intersectors.add<CudaSphereModel, PointModel,        DiscreteIntersection, true>(this);
        intersectors.add<CudaSphereModel, CudaSphereModel,   DiscreteIntersection, false>(this);
        //intersectors.add<LineModel,       CudaSphereModel,   CudaProximityIntersection, true>(this);
        intersectors.add<TriangleModel,   CudaSphereModel,   CudaProximityIntersection, true>(this);
    }

};


int CudaProximityIntersectionClass = core::RegisterObject("TODO-CudaProximityIntersection")
        .add< CudaProximityIntersection >()
        ;

sofa::helper::Creator<core::componentmodel::collision::Contact::Factory, component::collision::RayContact<CudaSphereModel> > RayCudaSphereContactClass("default",true);
sofa::helper::Creator<core::componentmodel::collision::Contact::Factory, component::collision::RayContact<CudaSphereModel> > RayCudaSphereContactClass2("LagrangianMultiplier",true);
sofa::helper::Creator<core::componentmodel::collision::Contact::Factory, component::collision::RayContact<CudaSphereModel> > RayCudaSphereContactClass3("FrictionContact",true);


sofa::helper::Creator<sofa::core::componentmodel::collision::Contact::Factory, sofa::component::collision::BarycentricPenalityContact<CudaRigidDistanceGridCollisionModel, CudaRigidDistanceGridCollisionModel> > CudaDistanceGridCudaDistanceGridContactClass("default", true);
sofa::helper::Creator<sofa::core::componentmodel::collision::Contact::Factory, sofa::component::collision::BarycentricPenalityContact<CudaRigidDistanceGridCollisionModel, sofa::component::collision::RigidDistanceGridCollisionModel> > CudaDistanceGridDistanceGridContactClass("default", true);
sofa::helper::Creator<sofa::core::componentmodel::collision::Contact::Factory, sofa::component::collision::BarycentricPenalityContact<CudaRigidDistanceGridCollisionModel, sofa::component::collision::PointModel> > CudaDistanceGridPointContactClass("default", true);
sofa::helper::Creator<sofa::core::componentmodel::collision::Contact::Factory, sofa::component::collision::BarycentricPenalityContact<CudaRigidDistanceGridCollisionModel, sofa::component::collision::SphereModel> > CudaDistanceGridSphereContactClass("default", true);
sofa::helper::Creator<sofa::core::componentmodel::collision::Contact::Factory, sofa::component::collision::BarycentricPenalityContact<CudaRigidDistanceGridCollisionModel, sofa::component::collision::TriangleModel> > CudaDistanceGridTriangleContactClass("default", true);

} // namespace cuda

} // namespace gpu

} // namespace sofa
