#include "GlPickedPoint.h"


GlPickedPoint::GlPickedPoint( BaseMechanicalState::SPtr pState, nat index, Vec3 origin, Vec3 pickedLocation, int x, int y )
{
    state = pState; assert(state);
    index = index;
//    point = pickedLocation;
    distance = (pickedLocation-origin).norm();
//    cur_x = x;
//    cur_x = y;
//    m_anchor = NULL;
}

//GlPickedPoint::~GlPickedPoint(){
//    if( m_anchor )
//        delete m_anchor;
//}

//void GlPickedPoint::setAnchor( Anchor* a ){ m_anchor = a; }

////GlPickedPoint GlPickedPoint::pick(GLdouble ox, GLdouble oy, GLdouble oz, int x, int y, const GLdouble* mvmatrix, const GLdouble* projmatrix, const GLint* viewport)
////{

////}


//void GlPickedPoint::move( int x, int y, const GLdouble* mvmatrix, const GLdouble* projmatrix, const GLint* viewport )
//{
//    GLdouble p[3];
//    gluUnProject ( x, viewport[3]-y-1, distance, mvmatrix, projmatrix, viewport, &p[0], &p[1], &p[2]); // new position of the picked point

//    m_anchor->move(  Vec3(p[0],p[1],p[2]) );
//}

//BaseMechanicalState* GlPickedPoint::getState() { return pickedState.get(); }
//nat GlPickedPoint::getIndex() const { return pickedIndex; }
