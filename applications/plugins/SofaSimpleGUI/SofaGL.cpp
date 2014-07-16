#include "SofaGL.h"

namespace sofa {
namespace newgui {

template <typename T> inline T sqr(const T& t){ return t*t; }


SofaGL::SofaGL(SofaScene *s)
{
    _vparams = sofa::core::visual::VisualParams::defaultInstance();
    _vparams->drawTool() = &_drawToolGL;
    _sofaScene = s;
}

void SofaGL::init(){}

void SofaGL::draw()
{
    //                cout<<"SofaGL::draw" << endl;
    //                sofaScene->printScene();
    glGetIntegerv (GL_VIEWPORT, _viewport);
    glGetDoublev (GL_MODELVIEW_MATRIX, _mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, _projmatrix);

    sofa::simulation::getSimulation()->updateVisual(_sofaScene->sroot().get()); // needed to update normals ! (i think it should be better if updateVisual() was called from draw(), why it is not already the case ?)
    sofa::simulation::getSimulation()->draw(_vparams,_sofaScene->sroot().get());
}

void SofaGL::getPickDirection( GLdouble* dx, GLdouble* dy, GLdouble* dz, int x, int y )
{
    // Intersection of the ray with the near and far planes
    GLint realy = _viewport[3] - (GLint) y - 1; // convert coordinates from image space (y downward) to window space (y upward)
    GLdouble wx, wy, wz;  /*  returned world x, y, z coords  */
    gluUnProject ((GLdouble) x, (GLdouble) realy, 0.0, _mvmatrix, _projmatrix, _viewport, &wx, &wy, &wz); // z=0: near plane
    //cout<<"World coords at z=0.0 are ("<<wx<<","<<wy<<","<<wz<<")"<<endl;
    GLdouble wx1, wy1, wz1;
    gluUnProject ((GLdouble) x, (GLdouble) realy, 1.0, _mvmatrix, _projmatrix, _viewport, &wx1, &wy1, &wz1); // z=1: far plane

    GLdouble nrm = sqrt( sqr(wx1-wx) + sqr(wy1-wy) + sqr(wz1-wz) );
    *dx = (wx1-wx)/nrm;
    *dy = (wy1-wy)/nrm;
    *dz = (wz1-wz)/nrm;

}


PickedPoint SofaGL::pick(GLdouble ox, GLdouble oy, GLdouble oz, int x, int y )
{
    Vec3 origin(ox,oy,oz), direction;
    getPickDirection(&direction[0],&direction[1],&direction[2],x,y);

    double distance = 10.5, distanceGrowth = 0.1; // cone around the ray ????
    //    cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth );
    picker.execute( _sofaScene->sroot()->getContext() );

    PickedPoint pickedPoint;
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


Interactor* SofaGL::getInteractor( const PickedPoint& glpicked )
{
    cout << "SofaGL::getInteractor, looking for " << glpicked << endl;
    for( Picked_to_Interactor::iterator i=_picked_to_interactor.begin(); i!=_picked_to_interactor.end(); i++ )
    {
        cout << "SofaGL::getInteractor, map contains " << (*i).first << endl;
    }
    if( _picked_to_interactor.find(glpicked)!=_picked_to_interactor.end() ) // there is already an interactor on this particle
    {
        return _picked_to_interactor[glpicked];
    }
    else {                                             // new interactor
        return NULL;
    }
}



Interactor* SofaGL::pickInteractor( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y )
{

    Vec3 origin(ox,oy,oz), direction;
    getPickDirection(&direction[0],&direction[1],&direction[2],x,y);
    double distance = 10.5, distanceGrowth = 0.1; // cone around the ray ????
//    cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth, Tag("!NoPicking") );
    picker.execute(_sofaScene->sroot()->getContext());

    if (!picker.particles.empty())
    {
        PickedPoint pickedPoint(picker.particles.begin()->second.first, picker.particles.begin()->second.second);
        if( _picked_to_interactor.find(pickedPoint)!=_picked_to_interactor.end() )
            return _picked_to_interactor[pickedPoint];
    }

    return NULL;
}


void SofaGL::attach( Interactor* interactor )
{
    interactor->attach( _sofaScene );
    _picked_to_interactor[interactor->getPickedPoint()] = interactor;
    //    cout<<"SofaGL::attach "<< endl; _sofaScene->printGraph();
}

void SofaGL::move( Interactor* interactor, int x, int y)
{
    if( !interactor )
        return;

    // get the distance to the current point
    Vec3 current = interactor->getPoint();
    GLdouble wcur[3]; // window coordinates of the current point
    gluProject(current[0],current[1],current[2],_mvmatrix,_projmatrix,_viewport,wcur,wcur+1,wcur+2);
    //        cout << "current point = " << current << endl;
    //        cout<<"move anchor, distance = " << wcur[2] << endl;

    // compute and set the position of the new point
    GLdouble p[3];
    gluUnProject ( x, _viewport[3]-y-1, wcur[2], _mvmatrix, _projmatrix, _viewport, &p[0], &p[1], &p[2]); // new position of the picked point
    //        cout<<"x="<< x <<", y="<< y <<", X="<<p[0]<<", Y="<<p[1]<<", Z="<<p[2]<<endl;
    interactor->setPoint(Vec3(p[0], p[1], p[2]));
}

void SofaGL::detach( Interactor* drag)
{
    if( !drag )
        return;

    // remove it from the map
    Picked_to_Interactor::iterator i=_picked_to_interactor.begin();
    while( i!=_picked_to_interactor.end() && (*i).second != drag )
        i++;
    if( i!=_picked_to_interactor.end() ){
        //                cout << "Deleted interactor at " << (*i).first << endl;
        _picked_to_interactor.erase(i);
        //                cout << "new count of interactors: " << picked_to_interactor.size() << endl;
    }
    else assert( false && "Active interactor not found in the map" );

    drag->detach();
    //    cout<<"SofaGL::detach "<< endl; _sofaScene->printGraph();
}


void SofaGL::viewAll( SReal* xcam, SReal* ycam, SReal* zcam, SReal* xcen, SReal* ycen, SReal* zcen, SReal a, SReal* nearPlane, SReal* farPlane)
{
    // scene center and radius
    SReal xmin, xmax, ymin, ymax, zmin, zmax;
    _sofaScene->getBoundingBox(&xmin,&xmax,&ymin,&ymax,&zmin,&zmax);
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
    *nearPlane = distance - radius*1.5;
    *farPlane  = distance + radius*1.5;
}


}//newgui
}//sofa
