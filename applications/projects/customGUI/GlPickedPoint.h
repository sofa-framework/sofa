#ifndef GLPICKEDPOINT_H
#define GLPICKEDPOINT_H
#include <iostream>
#include <sofa/core/behavior/BaseMechanicalState.h>
using sofa::core::behavior::BaseMechanicalState;

typedef std::size_t nat;
typedef sofa::defaulttype::Vector3 Vec3;


class GlPickedPoint
{
public:
    BaseMechanicalState::SPtr state;
    nat index;
    SReal distance; ///< distance from camera
    operator bool() const { return state != NULL; }


    GlPickedPoint(BaseMechanicalState::SPtr pState, nat index, Vec3 origin, Vec3 pickedLocation, int x, int y );
    ~GlPickedPoint();


    inline friend std::ostream& operator << ( std::ostream& out, const GlPickedPoint p){
        out << "state: " << p.state->getName() << ", index: " << p.index << ", distance: " << p.distance;
        return out;
    }


};

#endif // GLPICKEDPOINT_H
