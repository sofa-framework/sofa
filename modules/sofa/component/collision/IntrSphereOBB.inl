#ifndef INTRSPHEREOBB_INL
#define INTRSPHEREOBB_INL
#include <sofa/component/collision/IntrSphereOBB.h>

namespace sofa{
namespace component{
namespace collision{

template <class TDataTypes,class TDataTypes2>
TIntrSphereOBB<TDataTypes,TDataTypes2>::TIntrSphereOBB (const IntrSph& sphere, const Box & box) : _sph(&sphere),mBox(&box){this->_is_colliding = false;}

template <class TDataTypes,class TDataTypes2>
bool TIntrSphereOBB<TDataTypes,TDataTypes2>::Find(){
    _is_colliding = true;

    _pt_on_second = mBox->center();
    _pt_on_first = _sph->center();
    Vec<3,Real> centeredPt = _pt_on_first - _pt_on_second;

    //projecting the center of the sphere on the OBB
    Real coord_i;
    for(int i = 0 ; i < 3 ; ++i){
        coord_i = mBox->axis(i) * centeredPt;//the i-th coordinate of the sphere center in the OBB's local frame

        if(coord_i < -mBox->extent(i)){//if the i-th coordinate is less than -mBox->extent(i), we are outside of the OBB by the negative sens, so the i-th nearest coordinate to the sphere center
                                       //coordinate is -mBox->extent
            _is_colliding = false;
            coord_i = -mBox->extent(i);
        }
        else if(coord_i > mBox->extent(i)){//same idea in the positive sens
            _is_colliding = false;
            coord_i = mBox->extent(i);
        }

        _pt_on_second += coord_i * mBox->axis(i);
    }

    //The normal response which _sep_axis have the same direction than the one which goes from the sphere center to the OBB contact point.
    //If the sphere and the OBB are colliding
    _sep_axis = _pt_on_second - _pt_on_first;
    IntrUtil<Real>::normalize(_sep_axis);

    _pt_on_first += _sph->r() * _sep_axis;

    if((!_is_colliding) && ((_pt_on_second - _pt_on_first) * centeredPt > 0))
        _is_colliding = true;

    return true;
}

template <typename TDataTypes,typename TDataTypes2>
typename TIntrSphereOBB<TDataTypes,TDataTypes2>::Real TIntrSphereOBB<TDataTypes,TDataTypes2>::distance()const{
    return fabs(_sep_axis * (_pt_on_first - _pt_on_second));
}


}
}
}

#endif // INTRSPHEREOBB_INL
