#include "DragAnchor.h"
#include "GlPickedPoint.h"

DragAnchor::DragAnchor(GlPickedPoint *picked)
    : Anchor(picked)
{
}

void DragAnchor::move( const Vec3& p )
{
    MechanicalObject3::WriteVecCoord xanchor = anchorDof->writePositions();
    xanchor[0] = p;
}
