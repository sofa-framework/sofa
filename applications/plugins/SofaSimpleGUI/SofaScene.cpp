#include "SofaScene.h"
#include "Interactor.h"
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/init.h>
#include <sofa/simulation/common/xml/initXml.h>

// sofa types should not be exposed
//typedef sofa::defaulttype::Vector3 Vec3;
//typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


namespace sofa {
namespace newgui {


typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


SofaScene::SofaScene()
{
	sofa::core::ExecParams::defaultInstance()->setAspectID(0);
	boost::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel = 0;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

    sofa::simulation::setSimulation(new SofaSimulation());

    sofa::component::init();
    sofa::simulation::xml::initXml();

    _groot = sofa::simulation::getSimulation()->createNewGraph("");
    _groot->setName("theRoot");
}

void SofaScene::step( SReal dt)
{
    sofa::simulation::getSimulation()->animate(_groot.get(),dt);
}

void SofaScene::printGraph()
{
    sofa::simulation::getSimulation()->print(_groot.get());
}


void SofaScene::init(std::vector<std::string> plugins, const std::string& fileName )
{

    // --- plugins ---
    for (unsigned int i=0; i<plugins.size(); i++){
        sout<<"SofaScene::init, loading plugin " << plugins[i] << sendl;
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);
    }

    sofa::helper::system::PluginManager::getInstance().init();


    // --- Create simulation graph ---
    if(fileName.empty())
        sout << "SofaGLScene::init, no file to load" << sendl;
    else open(fileName.c_str() );

}

void SofaScene::reset()
{
    SofaSimulation::reset(_groot.get());
}

void SofaScene::open(const char *filename)
{
    if(_sroot){
        unload(_sroot);
        unload(_iroot);
    }
    _sroot = _groot->createChild("sroot");
    _iroot = _groot->createChild("iroot");

    Node::SPtr loadroot = load( filename );
    if( !loadroot ){
        cerr << "loading failed" << endl;
        return;
    }
    _currentFileName = filename;


    _sroot->addChild(loadroot);


    SofaSimulation::init(_groot.get());
//    cout<<"SofaScene::init, scene loaded" << endl;
//    printGraph();
}

void SofaScene::getBoundingBox( SReal* xmin, SReal* xmax, SReal* ymin, SReal* ymax, SReal* zmin, SReal* zmax )
{
    SReal pmin[3], pmax[3];
    computeBBox( _groot.get(), pmin, pmax );
    *xmin = pmin[0]; *xmax = pmax[0];
    *ymin = pmin[1]; *ymax = pmax[1];
    *zmin = pmin[2]; *zmax = pmax[2];
}

void SofaScene::insertInteractor( Interactor * interactor )
{
    _iroot->addChild(interactor->getNode());
}




}// newgui
}// sofa
