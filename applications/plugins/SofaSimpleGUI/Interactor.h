#ifndef SOFA_Interactor_H
#define SOFA_Interactor_H

#include <SofaSimpleGUI/config.h>
#include <sofa/simulation/common/Node.h>
#include "PickedPoint.h"



namespace sofa{
namespace simplegui{

class SofaScene;

using simulation::Node;


/** @brief Base class for all interactors.
 * Interactors are sofa subgraphs used to apply actions to the scene, typically through user interaction.
 *
 * Note that they are not necessarily moved using the mouse, since there is no mouse or window coordinates in the API.
 * @author Francois Faure, 2014
*/
class SOFA_SOFASIMPLEGUI_API Interactor
{
protected:
    Node::SPtr _interactionNode;  ///< Scene node used to implement this
    PickedPoint _pickedPoint;     ///< The point attached to this

public:

    typedef defaulttype::Vector3 Vec3;

    Interactor( const PickedPoint&  picked );
    virtual ~Interactor();

    /// Insert this in the scene as a child of the given node. If overloaded, this function should be called at the beginning of the overloaded function.
    virtual void attach( SofaScene* );

    /// Remove this from the scene, without destroying it. If overloaded, this function should be called at the end of the overloaded function.
    virtual void detach();

    /// Current interaction point
    virtual Vec3 getPoint()=0;

    /// Displace the interaction to the given point
    virtual void setPoint( const Vec3& p ) = 0;

    /// Root of the interactor graph
    Node::SPtr getNode() { return _interactionNode; }

    /// Simulated point attached to this
    const PickedPoint& getPickedPoint() const { return _pickedPoint; }
};

}
}

#endif // SOFA_Interactor_H
