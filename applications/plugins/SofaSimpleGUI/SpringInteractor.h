#ifndef SOFA_SpringInteractor_H
#define SOFA_SpringInteractor_H

#include "initSimpleGUI.h"
#include "Interactor.h"

namespace sofa{
namespace simplegui{

/**
 * @brief Interaction using a spring.
 * An interactor, typically attached to the mouse pointer, pulls a control point using a spring.
 * @author Francois Faure, 2014
 */
class SOFA_SOFASIMPLEGUI_API SpringInteractor: public Interactor
{
    typedef Interactor Inherited;
protected:
    MechanicalObject3::SPtr _interactorDof;
    StiffSpringForceField3::SPtr _spring;
public:
    /**
     * @brief SpringInteractor
     * @param picked The picked point.
     * @param stiffness The stiffness of the spring attached to the picked point.
     */
    SpringInteractor(const PickedPoint&  picked, SReal stiffness=(SReal) 100.);

    /// Insert this in the scene as a child of the given node
    virtual void attach( SofaScene* scene );

    /// Remove this from the scene, without destroying it.
    virtual void detach();

    /// current interaction point
    Vec3 getPoint();

    /// Displace the interaction to the given point
    virtual void setPoint( const Vec3& p );

};

}
}

#endif // SOFA_SpringInteractor_H
