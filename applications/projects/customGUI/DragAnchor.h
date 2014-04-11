#ifndef SOFA_DragAnchor_H
#define SOFA_DragAnchor_H

#include "Anchor.h"

class DragAnchor: public Anchor
{
public:
    DragAnchor(GlPickedPoint*  picked);

    /// Displace the interaction to the given point
    virtual void move( const Vec3& p );

};

#endif // SOFA_DragAnchor_H
