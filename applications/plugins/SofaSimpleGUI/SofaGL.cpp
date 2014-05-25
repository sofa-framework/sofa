#include "SofaGL.h"

namespace sofa {
namespace newgui {



SofaGL::SofaGL(SofaScene *s)
{
    vparams = sofa::core::visual::VisualParams::defaultInstance();
    vparams->drawTool() = &drawToolGL;
    sofaScene = s;
}

void SofaGL::init(){}

void SofaGL::draw()
{
//                cout<<"SofaGL::draw" << endl;
//                sofaScene->printScene();
    glGetIntegerv (GL_VIEWPORT, viewport);
    glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);

    sofa::simulation::getSimulation()->updateVisual(sofaScene->sroot().get()); // needed to update normals ! (i think it should be better if updateVisual() was called from draw(), why it is not already the case ?)
    sofa::simulation::getSimulation()->draw(vparams,sofaScene->sroot().get());
}

PickedPoint SofaGL::pick( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y )
{
    PickedPoint pickedPoint;

    // Intersection of the ray with the near and far planes
    GLint realy = viewport[3] - (GLint) y - 1; // convert coordinates from image space (y downward) to window space (y upward)
    GLdouble wx, wy, wz;  /*  returned world x, y, z coords  */
    gluUnProject ((GLdouble) x, (GLdouble) realy, 0.0, mvmatrix, projmatrix, viewport, &wx, &wy, &wz); // z=0: near plane
    //cout<<"World coords at z=0.0 are ("<<wx<<","<<wy<<","<<wz<<")"<<endl;
    GLdouble wx1, wy1, wz1;
    gluUnProject ((GLdouble) x, (GLdouble) realy, 1.0, mvmatrix, projmatrix, viewport, &wx1, &wy1, &wz1); // z=1: far plane


    // Search for a particle in this direction
    Vec3 origin(ox,oy,oz), direction(wx1-wx, wy1-wy, wz1-wz);
    direction.normalize();
    double distance = 10.5, distanceGrowth = 0.1; // cone around the ray ????
//    cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth );
    picker.execute(sofaScene->sroot()->getContext());

    if (!picker.particles.empty())
    {
        sofa::core::behavior::BaseMechanicalState *mstate = picker.particles.begin()->second.first;
        unsigned index = picker.particles.begin()->second.second;

        pickedPoint.state = mstate;
        pickedPoint.index = index;
        pickedPoint.point = Vec3(mstate->getPX(index), mstate->getPY(index), mstate->getPZ(index));
    }

    return pickedPoint;
}

//Interactor* SofaGL::pickInteractor( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y )
//{
//    PickedPoint pickedPoint;

//    // Intersection of the ray with the near and far planes
//    GLint realy = viewport[3] - (GLint) y - 1; // convert coordinates from image space (y downward) to window space (y upward)
//    GLdouble wx, wy, wz;  /*  returned world x, y, z coords  */
//    gluUnProject ((GLdouble) x, (GLdouble) realy, 0.0, mvmatrix, projmatrix, viewport, &wx, &wy, &wz); // z=0: near plane
//    //cout<<"World coords at z=0.0 are ("<<wx<<","<<wy<<","<<wz<<")"<<endl;
//    GLdouble wx1, wy1, wz1;
//    gluUnProject ((GLdouble) x, (GLdouble) realy, 1.0, mvmatrix, projmatrix, viewport, &wx1, &wy1, &wz1); // z=1: far plane


//    // Search for a particle in this direction
//    Vec3 origin(ox,oy,oz), direction(wx1-wx, wy1-wy, wz1-wz);
//    direction.normalize();
//    double distance = 10.5, distanceGrowth = 0.1; // cone around the ray ????
////    cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
//    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth, Tag("!NoPicking") );
//    picker.execute(sofaScene->groot()->getContext());

//    if (!picker.particles.empty())
//    {
//        sofa::core::behavior::BaseMechanicalState *mstate = picker.particles.begin()->second.first;
//        unsigned index = picker.particles.begin()->second.second;

//        pickedPoint.state = mstate;
//        pickedPoint.index = index;
//        pickedPoint.point = Vec3(mstate->getPX(index), mstate->getPY(index), mstate->getPZ(index));
//    }

//    return pickedPoint;
//}

void SofaGL::attach( Interactor* interactor )
{
    interactor->attach( sofaScene );
//    sofaScene->insertInteractor( interactor );
}

void SofaGL::move( Interactor* interactor, int x, int y)
{
    if( !interactor )
        return;

    // get the distance to the current point
    Vec3 current = interactor->getPoint();
    GLdouble wcur[3]; // window coordinates of the current point
    gluProject(current[0],current[1],current[2],mvmatrix,projmatrix,viewport,wcur,wcur+1,wcur+2);
    //        cout << "current point = " << current << endl;
    //        cout<<"move anchor, distance = " << wcur[2] << endl;

    // compute and set the position of the new point
    GLdouble p[3];
    gluUnProject ( x, viewport[3]-y-1, wcur[2], mvmatrix, projmatrix, viewport, &p[0], &p[1], &p[2]); // new position of the picked point
    //        cout<<"x="<< x <<", y="<< y <<", X="<<p[0]<<", Y="<<p[1]<<", Z="<<p[2]<<endl;
    interactor->setPoint(Vec3(p[0], p[1], p[2]));
}

void SofaGL::detach( Interactor* interactor)
{
    if( !interactor )
        return;

    interactor->detach();
}

template <typename T> inline T sqr(const T& t){ return t*t; }

void SofaGL::viewAll( SReal* xcam, SReal* ycam, SReal* zcam, SReal* xcen, SReal* ycen, SReal* zcen, SReal a, SReal* near, SReal* far)
{
    // scene center and radius
    SReal xmin, xmax, ymin, ymax, zmin, zmax;
    sofaScene->getBoundingBox(&xmin,&xmax,&ymin,&ymax,&zmin,&zmax);
    *xcen = (xmin+xmax)*0.5;
    *ycen = (ymin+ymax)*0.5;
    *zcen = (zmin+zmax)*0.5;
    SReal radius = sqrt( sqr(xmin-xmax) + sqr(ymin-ymax) + sqr(zmin-zmax) )*0.5;

    // Desired distance:  distance * tan(a) = radius
    SReal distance = radius / tan(a);

    // move the camera along the current camera-center line, at the right distance
    // cam = cen + distance * (cam-cen)/|cam-cen|
    SReal curdist = sqrt( sqr(*xcam-*xcen)+sqr(*ycam-*ycen)+sqr(*zcam-*zcen) );
    *xcam = *xcen + distance * (*xcam-*xcen) / curdist;
    *ycam = *ycen + distance * (*ycam-*ycen) / curdist;
    *zcam = *zcen + distance * (*zcam-*zcen) / curdist;

    // update the depth bounds
    *near = distance - radius*1.5;
    *far  = distance + radius*1.5;
}


}//newgui
}//sofa
