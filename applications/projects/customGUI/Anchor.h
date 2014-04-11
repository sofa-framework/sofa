#ifndef SOFA_Anchor_H
#define SOFA_Anchor_H

#include <sofa/simulation/common/Node.h>
#include <sofa/component/typedef/Sofa_typedef.h>


using namespace sofa;
using simulation::Node;
class GlPickedPoint;
typedef sofa::defaulttype::Vector3 Vec3;


/** Base class for all anchors.
 * Anchors are sofa subgraphs representing interaction objects.
 * They can be controled by the user interface through a picked point.
*/
class Anchor
{
protected:
    Node::SPtr interactionNode;  ///< node representing this, set as child to parentNode
    MechanicalObject3::SPtr anchorDof;

public:
    Anchor( GlPickedPoint*  picked );

    virtual ~Anchor();

    void attach( Node::SPtr parent );

    /// Displace the interaction to the given point
    virtual void move( const Vec3& p ) = 0;

    void detach();
};

#endif // SOFA_Anchor_H
