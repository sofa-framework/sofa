#ifndef GLPICKEDPOINT_H
#define GLPICKEDPOINT_H

#include "initSimpleGUI.h"
#include <iostream>
#include <sofa/core/behavior/BaseMechanicalState.h>

using sofa::core::behavior::BaseMechanicalState;
typedef std::size_t nat;
typedef sofa::defaulttype::Vector3 Vec3;

namespace sofa{
namespace simplegui{

/**
 * @brief The PickedPoint struct represents a vertex of a State, typically picked using the mouse.
 * It is returned by the Sofa interface to the user application to set up an interaction.
 * We call it vaild if it corresponds to a valid index of an existing mechanical state, and invalid if not so (the pointer to the mechanical state is null)
 * @author Francois Faure, 2014
 */
struct SOFA_SOFASIMPLEGUI_API PickedPoint
{
    BaseMechanicalState::SPtr state; ///< the DOF of the picked object
    nat index;                       ///< index of the particle picked
    Vec3 point;                      ///< location of the picked particle in world space

    /// Conversion to boolean for easier test writing. True iff the PickedPoint is valid. Default value is converted to false.
    operator bool() const { return state != NULL; }

    PickedPoint(BaseMechanicalState::SPtr state=0, nat index=0)
        : state(state)
        , index(index)
    {    }

    inline friend std::ostream& operator << ( std::ostream& out, const PickedPoint p){
        out << "state: " << p.state->getName() << ", index: " << p.index << ", point: " << p.point;
        return out;
    }

    /// Comparison operator used in maps
    bool operator < (const PickedPoint& p ) const {
        return state < p.state || index < p.index;
    }

    /// Comparison operator used in maps
    bool operator != (const PickedPoint& p ) const {
        return *this<p || p<*this;
    }

    /// Comparison operator used in maps
    bool operator == (const PickedPoint& p ) const {
        return ! *this!=p;
    }




};

}
}

#endif // GLPICKEDPOINT_H
