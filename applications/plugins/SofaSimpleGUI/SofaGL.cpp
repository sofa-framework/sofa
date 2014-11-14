#include <GL/glew.h>
#include "SofaGL.h"
#include "VisualPickVisitor.h"
#include <sofa/core/objectmodel/Tag.h>

namespace sofa {
using core::objectmodel::Tag;

namespace simplegui {

template <typename T> inline T sqr(const T& t){ return t*t; }


SofaGL::SofaGL(SofaScene *s) :
    _sofaScene(s)
{
    if(!_sofaScene)
    {
        std::cerr << "Error: you are trying to create a SofaGL object with a null SofaScene" << std::endl;
        return;
    }

    glewInit();

    _vparams = sofa::core::visual::VisualParams::defaultInstance();
    _vparams->drawTool() = &_drawToolGL;
    _vparams->setSupported(sofa::core::visual::API_OpenGL);

    _isPicking = false;


    _sofaScene->initVisual();
}

void SofaGL::draw()
{
    glGetIntegerv (GL_VIEWPORT, _viewport);
    glGetDoublev (GL_MODELVIEW_MATRIX, _mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, _projmatrix);

    if(_vparams)
    {
        _vparams->viewport() = sofa::helper::fixed_array<int, 4>(_viewport[0], _viewport[1], _viewport[2], _viewport[3]);
        SReal xmin,xmax,ymin,ymax,zmin,zmax;
        _sofaScene->getBoundingBox(&xmin,&xmax,&ymin,&ymax,&zmin,&zmax);
        _vparams->sceneBBox() = sofa::defaulttype::BoundingBox(xmin,xmax,ymin,ymax,zmin,zmax);
        _vparams->setProjectionMatrix(_projmatrix);
        _vparams->setModelViewMatrix(_mvmatrix);
    }

    //_sofaScene->getSimulation()->updateVisual(_sofaScene->getSimulation()->GetRoot().get()); // needed to update normals and VBOs ! (i think it should be better if updateVisual() was called from draw(), why it is not already the case ?)
    _sofaScene->updateVisual(); // needed to update normals and VBOs ! (i think it should be better if updateVisual() was called from draw(), why it is not already the case ?)

    if( _isPicking ){

        // start picking
        glSelectBuffer(BUFSIZE,selectBuf);
        glRenderMode(GL_SELECT);

        glInitNames();

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();

        gluPickMatrix(pickX,_viewport[3]-pickY,5,5,_viewport);
        glMultMatrixd(_projmatrix);
        glMatrixMode(GL_MODELVIEW);

        // draw
        _vparams->pass() = sofa::core::visual::VisualParams::Std;
        VisualPickVisitor pick ( _vparams );
        pick.setTags(_sofaScene->groot()->getTags());
        cerr<<"SofaGL::draw root used " <<  endl;
        _sofaScene->groot()->execute ( &pick );

        // stop picking
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glFlush();
        hits = glRenderMode(GL_RENDER);
        if (hits != 0)
        {
            GLuint* buffer = selectBuf;
            // process the hits
            GLint i, j, numberOfNames;
            GLuint names, *ptr, minZ,*ptrNames;

            ptr = (GLuint *) buffer;
            minZ = 0xffffffff;
            for (i = 0; i < hits; i++) {
                names = *ptr;
                ptr++;
                if (*ptr < minZ) {
                    numberOfNames = names;
                    minZ = *ptr;
                    ptrNames = ptr+2;
                }

                ptr += names+2;
            }
            if (numberOfNames > 0) {
                cerr << "You picked object  ";
                ptr = ptrNames;
                for (j = 0; j < numberOfNames; j++,ptr++) {
                    cerr<< pick.names[*ptr] << " ";
                }
            }
            else
                cerr<<"You didn't click a snowman!";
            cerr<<endl;
        }
        else cerr<<"no hits !" << endl;
        _isPicking = false;

    }

//    _sofaScene->getSimulation()->draw(_vparams, _sofaScene->getSimulation()->GetRoot().get());
    draw(_vparams);
}

void SofaGL::draw(sofa::core::visual::VisualParams* vparams)
{
    core::visual::VisualLoop* vloop = _sofaScene->groot()->getVisualLoop();
    assert(vloop);
    if (!vparams) vparams = sofa::core::visual::VisualParams::defaultInstance();
    vparams->update();
    vloop->drawStep(vparams);
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


void SofaGL::glPick(int x, int y )
{
    pickX = x; pickY = y;
    _isPicking = true;
}

PickedPoint SofaGL::pick(GLdouble ox, GLdouble oy, GLdouble oz, int x, int y )
{
    Vec3 origin(ox,oy,oz), direction;
    getPickDirection(&direction[0],&direction[1],&direction[2],x,y);

    double distance = 10.5, distanceGrowth = 0.1; // cone around the ray ????
    //    cout<< "SofaGL::rayPick from origin " << origin << ", in direction " << direction << endl;
    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth );
    picker.execute( _sofaScene->groot()->getContext() );

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
    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth, sofa::core::objectmodel::Tag("!NoPicking") );
    picker.execute(_sofaScene->groot()->getContext());

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

void SofaGL::getSceneBBox( float* xmin, float* ymin, float* zmin, float* xmax, float* ymax, float* zmax )
{
    SReal xm, xM, ym, yM, zm, zM;
    _sofaScene->getBoundingBox(&xm,&xM,&ym,&yM,&zm,&zM);
    //    cerr << "SofaGL::getSceneBBox, xm=" << xm <<", xM=" << xM << endl;
    *xmin=xm, *xmax=xM, *ymin=ym, *ymax=yM, *zmin=zm, *zmax=zM;
}


void SofaGL::viewAll( SReal* xcam, SReal* ycam, SReal* zcam, SReal* xcen, SReal* ycen, SReal* zcen, SReal a, SReal* nearPlane, SReal* farPlane)
{
    // scene center and radius
    SReal xmin, xmax, ymin, ymax, zmin, zmax;
    _sofaScene->getBoundingBox(&xmin,&xmax,&ymin,&ymax,&zmin,&zmax);
    cout<<"SofaGL::viewAll, bounding box = ("<< xmin <<" "<<ymin<<" "<<zmin<<"),("<<xmax<<" "<<ymax<<" "<<zmax<<")"<<endl;
    *xcen = (xmin+xmax)*0.5;
    *ycen = (ymin+ymax)*0.5;
    *zcen = (zmin+zmax)*0.5;
    SReal radius = sqrt( sqr(xmin-xmax) + sqr(ymin-ymax) + sqr(zmin-zmax) );

    // Desired distance:  distance * tan(a) = radius
    SReal distance = 2 * radius / tan(a);
    //    SReal ratio = ((SReal) _viewport[3] - _viewport[1])/(_viewport[2] - _viewport[0]);
    //    distance *= ratio;
    cout<<"SofaGL::viewAll, angle = " << a << ", tan = " << tan(a) << ", distance = " << distance << endl;
    cout<<"SofaGL::viewAll, xmin xmax ymin ymax zmin zmax = " << xmin << " " << xmax <<" "<<ymin<<" "<<ymax<<" "<<zmin<<" "<<zmax<< endl;

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
