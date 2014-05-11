#include "SofaScene.h"

#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/init.h>
#include <sofa/simulation/common/xml/initXml.h>

namespace sofa {
namespace newgui {


typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


SofaScene::SofaScene()
{
    sofa::simulation::setSimulation(new SofaSimulation());

    sofa::component::init();
    sofa::simulation::xml::initXml();

    _groot = sofa::simulation::getSimulation()->createNewGraph("");
    _groot->setName("theRoot");
}

void SofaScene::step( SReal dt)
{
    //        if( debug )
//                cout<<"SofaScene::step" << endl;
    sofa::simulation::getSimulation()->animate(_groot.get(),dt);
}

Node::SPtr SofaScene::groot() { return _groot; }

void SofaScene::printScene()
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
    _sroot = sofa::simulation::getSimulation()->load(fileName.c_str());
    if (_sroot!=NULL)
    {
        _sroot->setName("sceneRoot");
        _groot->addChild(_sroot);
    }
    else {
        serr << "SofaScene::init, could not load scene " << fileName << ", is the path ok ?" << sendl;
    }

    SofaSimulation::init(_groot.get());

//    if( debug ){
//        cout<<"SofaScene::init, scene loaded" << endl;
//        sofa::simulation::getSimulation()->print(groot.get());
//    }

}



}// newgui
}// sofa
