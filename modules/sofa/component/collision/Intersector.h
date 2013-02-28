#ifndef INTERSECTOR_H
#define INTERSECTOR_H

#include <sofa/defaulttype/Vec.h>

namespace sofa{
namespace component{
namespace collision{


template <class Real>
class Intersector{
public:
    typedef sofa::defaulttype::Vec<3,Real> Vec3;

    inline const Vec3 & separatingAxis()const{
        return _sep_axis;
    }

    inline const Vec3 & pointOnFirst()const{
        return _pt_on_first;
    }

    inline const Vec3 & pointOnSecond()const{
        return _pt_on_second;
    }

    inline bool colliding()const{
        return _is_colliding;
    }
protected:
    Vec3 _sep_axis;
    Vec3 _pt_on_first;
    Vec3 _pt_on_second;
    Real mContactTime;
    bool _is_colliding;
};

}
}
}
#endif // INTERSECTOR_H
