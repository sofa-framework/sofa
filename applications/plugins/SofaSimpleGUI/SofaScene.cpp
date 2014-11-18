#include "SofaScene.h"
#include "Interactor.h"
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sofa/helper/system/PluginManager.h>
#include <SofaComponentMain/init.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/graph/DAGSimulation.h>

// sofa types should not be exposed
//typedef sofa::defaulttype::Vector3 Vec3;
//typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;
typedef sofa::simulation::graph::DAGSimulation SofaSimulation;


namespace sofa {
namespace simplegui {


typedef sofa::defaulttype::Vector3 Vec3;
typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


SofaScene::SofaScene()
{
    _groot = _iroot = NULL;
	sofa::core::ExecParams::defaultInstance()->setAspectID(0);
    boost::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel;// = NULL;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

    sofaSimulation = sofa::simulation::graph::getSimulation(); // creates one if it is not already created

    sofa::component::init();
    sofa::simulation::xml::initXml();
}

void SofaScene::step( SReal dt)
{
    sofaSimulation->animate(_groot,dt);
}

void SofaScene::printGraph()
{
    sofaSimulation->print(_groot);
}

void SofaScene::loadPlugins( std::vector<std::string> plugins )
{
    for (unsigned int i=0; i<plugins.size(); i++){
        cout<<"SofaScene::init, loading plugin " << plugins[i] << endl;
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);
    }

    sofa::helper::system::PluginManager::getInstance().init();
}

void SofaScene::open(const std::string& fileName )
{
    // --- Create simulation graph ---
    assert( !fileName.empty());

    if(_groot) sofaSimulation->unload (_groot);
    _groot = sofaSimulation->load( fileName.c_str() ).get();
    if(!_groot)
    {
        cerr << "loading failed" << endl;
        return;
    }

    _iroot = _groot->createChild("iroot").get();

//    _currentFileName = fileName;

    sofaSimulation->init(_groot);

    printGraph();
    SReal xm,xM,ym,yM,zm,zM;
    getBoundingBox(&xm,&xM,&ym,&yM,&zm,&zM);
    cout<<"SofaScene::setScene, xm="<<xm<<", xM"<< xM<< ", ym="<< ym<<", yM="<< yM<<", zm="<< zm<<", zM="<< zM<<endl;

}

void SofaScene::setScene(simulation::Node *node )
{
    if(_groot) sofaSimulation->unload (_groot);
    _groot = sofaSimulation->createNewGraph("root").get();
    _groot->addChild(node);
    _iroot = _groot->createChild("iroot").get();
    sofaSimulation->init(_groot);
}

void SofaScene::reset()
{
    sofaSimulation->reset(_groot);
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

//    sofaSimulation->::init(_groot);
////    cout<<"SofaScene::init, scene loaded" << endl;
////    printGraph();
//}

void SofaScene::getBoundingBox( SReal* xmin, SReal* xmax, SReal* ymin, SReal* ymax, SReal* zmin, SReal* zmax )
{
    SReal pmin[3], pmax[3];
    sofaSimulation->computeTotalBBox( _groot, pmin, pmax );
    *xmin = pmin[0]; *xmax = pmax[0];
    *ymin = pmin[1]; *ymax = pmax[1];
    *zmin = pmin[2]; *zmax = pmax[2];
}

void SofaScene::insertInteractor( Interactor * interactor )
{
	if(_iroot)
	    _iroot->addChild(interactor->getNode());
}

simulation::Node* SofaScene::groot() { return sofaSimulation->GetRoot().get(); }

void SofaScene::initVisual(){
    sofaSimulation->initTextures(sofaSimulation->GetRoot().get());
}

void SofaScene::updateVisual()
{
    sofaSimulation->updateVisual(sofaSimulation->GetRoot().get()); // needed to update normals and VBOs ! (i think it should be better if updateVisual() was called from draw(), why it is not already the case ?)
}

//void SofaScene::draw( sofa::core::visual::VisualParams* v)
//{
//    if( v==NULL )
//    {
//
//    }
//    sofaSimulation->draw(_vparams, groot() );
//}


}// newgui
}// sofa
