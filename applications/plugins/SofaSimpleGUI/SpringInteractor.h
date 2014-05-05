#ifndef SOFA_SpringInteractor_H
#define SOFA_SpringInteractor_H

#include "initSimpleGUI.h"
#include "Interactor.h"

namespace sofa{
namespace newgui{

/**
 * @brief Interaction using a spring.
 * @author Francois Faure, 2014
 */
class SOFA_SOFASIMPLEGUI_API SpringInteractor: public Interactor
{
protected:
    MechanicalObject3::SPtr anchorDof;
public:
    SpringInteractor(const PickedPoint&  picked);

    /// current interaction point
    Vec3 getPoint();

    /// Displace the interaction to the given point
    virtual void setPoint( const Vec3& p );

};

}
}

#endif // SOFA_SpringInteractor_H
