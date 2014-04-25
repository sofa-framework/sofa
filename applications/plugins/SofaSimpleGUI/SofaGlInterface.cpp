#include "SofaGlInterface.h"
#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/init.h>
#include <sofa/simulation/common/xml/initXml.h>

namespace sofa {
namespace newgui {


typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


SofaGlInterface::SofaGlInterface()
{
    debug = true;

    sofa::simulation::setSimulation(new ParentSimulation());

    sofa::component::init();
    sofa::simulation::xml::initXml();

    vparams = sofa::core::visual::VisualParams::defaultInstance();
    vparams->drawTool() = &drawToolGL;

    groot = sofa::simulation::getSimulation()->createNewGraph("");
    groot->setName("theRoot");
}

void SofaGlInterface::init( std::vector<std::string>& plugins, const std::string& fileName )
{

    // --- plugins ---
    for (unsigned int i=0; i<plugins.size(); i++){
        cout<<"SofaScene::init, loading plugin " << plugins[i] << endl;
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);
    }

    sofa::helper::system::PluginManager::getInstance().init();


    // --- Create simulation graph ---
    sroot = sofa::simulation::getSimulation()->load(fileName.c_str());
    if (sroot!=NULL)
    {
        sroot->setName("sceneRoot");
        groot->addChild(sroot);
    }
    else {
        serr << "SofaScene::init, could not load scene " << fileName << sendl;
    }

    ParentSimulation::init(groot.get());

    if( debug ){
        cout<<"SofaScene::init, scene loaded" << endl;
        sofa::simulation::getSimulation()->print(groot.get());
    }

}

void SofaGlInterface::printScene()
{
    sofa::simulation::getSimulation()->print(groot.get());
}

//void SofaGlInterface::reshape(int,int){}

void SofaGlInterface::glDraw()
{
    //        if(debug)
    //            cout<<"SofaScene::glDraw" << endl;
    glGetIntegerv (GL_VIEWPORT, viewport);
    glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
    sofa::simulation::getSimulation()->draw(vparams,groot.get());
}

void SofaGlInterface::animate()
{
    //        if( debug )
    //            cout<<"SofaScene::animate" << endl;
    sofa::simulation::getSimulation()->animate(groot.get(),0.04);
}

Node::SPtr SofaGlInterface::getRoot() { return groot; }

PickedPoint SofaGlInterface::pick( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y )
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
    if( debug ){
        cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
    }
    sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(), origin, direction, distance, distanceGrowth );
    picker.execute(groot->getContext());

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

void SofaGlInterface::attach( Interactor* interactor )
{
    interactor->attach(groot);
}

void SofaGlInterface::move( Interactor* interactor, int x, int y)
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

void SofaGlInterface::detach( Interactor* interactor)
{
    if( !interactor )
        return;

    interactor->detach();
}





}// newgui
}// sofa

