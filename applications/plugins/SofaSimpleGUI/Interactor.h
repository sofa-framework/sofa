#ifndef SOFA_Interactor_H
#define SOFA_Interactor_H

#include "initSimpleGUI.h"
#include <sofa/simulation/common/Node.h>
#include <sofa/component/typedef/Sofa_typedef.h>
#include "PickedPoint.h"


typedef sofa::defaulttype::Vector3 Vec3;

namespace sofa{
namespace newgui{

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
    Node::SPtr interactionNode;  ///< node representing this, set as child to a scene node

public:
    Interactor( const PickedPoint&  picked );
    virtual ~Interactor();

    /// Insert this in the scene as a child of the given node
    virtual void attach( Node::SPtr parent );

    /// Remove this from the scene, without destroying it.
    virtual void detach();

    /// Current interaction point
    virtual Vec3 getPoint()=0;

    /// Displace the interaction to the given point
    virtual void setPoint( const Vec3& p ) = 0;
};

}
}

#endif // SOFA_Interactor_H
