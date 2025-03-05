#pragma once

#include <sofa/component/collision/response/mapper/RigidContactMapper.h>
#include "BulletConvexHullModel.h"

namespace sofa::component::collision::response::mapper
{

template <class TVec3Types>
class ContactMapper<BulletConvexHullModel,TVec3Types > : public RigidContactMapper<BulletConvexHullModel, TVec3Types >{
public:
    typedef RigidContactMapper<BulletConvexHullModel, TVec3Types > Parent;

//I don't know why this is necessary but it is when i want to load this plugin, shit !
//    virtual typename Parent::MMechanicalState* createMapping(const char* name="contactPoints"){return Parent::createMapping(name);}

//    virtual void cleanup(){return Parent::cleanup();}

    int addPoint(const typename TVec3Types::Coord & P, int /*index*/,typename TVec3Types::Real & r)
    {
        r = 0;
        const typename TVec3Types::Coord & cP = P - this->model->center();
        const type::Quat<SReal> & ori = this->model->orientation();

        return RigidContactMapper<BulletConvexHullModel,TVec3Types >::addPoint(ori.inverseRotate(cP),0,r);
    }
};

#if  !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
extern template class SOFA_BULLETCOLLISIONDETECTION_API ContactMapper<BulletConvexHullModel,Vec3Types>;
#endif

} // namespace sofa::component::collision::response::mapper
