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
namespace simplegui {


typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


SofaScene::SofaScene()
{
	sofa::core::ExecParams::defaultInstance()->setAspectID(0);
    boost::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel;// = NULL;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

    sofa::simulation::setSimulation(new SofaSimulation());

    sofa::component::init();
    sofa::simulation::xml::initXml();
}

void SofaScene::step( SReal dt)
{
    sofa::simulation::getSimulation()->animate(_groot.get(),dt);
}

void SofaScene::printGraph()
{
    sofa::simulation::getSimulation()->print(_groot.get());
}

void SofaScene::loadPlugins( std::vector<std::string> plugins )
{
    for (unsigned int i=0; i<plugins.size(); i++){
        sout<<"SofaScene::init, loading plugin " << plugins[i] << sendl;
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);
    }

    sofa::helper::system::PluginManager::getInstance().init();
}

void SofaScene::open(const std::string& fileName )
{
    // --- Create simulation graph ---
    assert( !fileName.empty());

    if(_groot) unload (_groot);
    _groot = load( fileName.c_str() );
    if(!_groot)
    {
        cerr << "loading failed" << endl;
        return;
    }

    _iroot = _groot->createChild("iroot");

//    _currentFileName = fileName;

    SofaSimulation::init(_groot.get());

    printGraph();
    SReal xm,xM,ym,yM,zm,zM;
    getBoundingBox(&xm,&xM,&ym,&yM,&zm,&zM);
    cout<<"SofaScene::setScene, xm="<<xm<<", xM"<< xM<< ", ym="<< ym<<", yM="<< yM<<", zm="<< zm<<", zM="<< zM<<endl;

}

void SofaScene::setScene( Node::SPtr node )
{
    if(_groot) unload (_groot);
    _groot = sofa::simulation::getSimulation()->createNewGraph("root");
    _groot->addChild(node);
    _iroot = _groot->createChild("iroot");
    SofaSimulation::init(_groot.get());
}

void SofaScene::reset()
{
    SofaSimulation::reset(_groot.get());
}

//void SofaScene::open(const char *filename)
//{
//	unload(_groot);

//	_groot = load( filename );
//    if(!_groot)
//	{
//        cerr << "loading failed" << endl;
//        return;
//    }

//	_iroot = _groot->createChild("iroot");

//    _currentFileName = filename;

//    SofaSimulation::init(_groot.get());
////    cout<<"SofaScene::init, scene loaded" << endl;
////    printGraph();
//}

void SofaScene::getBoundingBox( SReal* xmin, SReal* xmax, SReal* ymin, SReal* ymax, SReal* zmin, SReal* zmax )
{
    SReal pmin[3], pmax[3];
    computeTotalBBox( _groot.get(), pmin, pmax );
    *xmin = pmin[0]; *xmax = pmax[0];
    *ymin = pmin[1]; *ymax = pmax[1];
    *zmin = pmin[2]; *zmax = pmax[2];
}

void SofaScene::insertInteractor( Interactor * interactor )
{
	if(_iroot)
	    _iroot->addChild(interactor->getNode());
}




}// newgui
}// sofa
