#ifndef SOFA_SpringInteractor_H
#define SOFA_SpringInteractor_H

#include <SofaSimpleGUI/config.h>
#include "Interactor.h"

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::simplegui
{
    using sofa::defaulttype::Vec3Types ;
    using MechanicalObject3 = sofa::component::statecontainer::MechanicalObject<Vec3Types> ;
    using StiffSpringForceField3 = sofa::component::solidmechanics::spring::StiffSpringForceField<Vec3Types> ;

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
    void attach( SofaScene* scene ) override;

    /// Remove this from the scene, without destroying it.
    void detach() override;

    /// current interaction point
    Vec3 getPoint() override;

    /// Displace the interaction to the given point
    void setPoint( const Vec3& p ) override;

};

}


#endif // SOFA_SpringInteractor_H
