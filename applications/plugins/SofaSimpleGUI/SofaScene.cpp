#include "SofaScene.h"
#include "Interactor.h"
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

// sofa types should not be exposed
//typedef sofa::type::Vec3 Vec3;
//typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;
typedef sofa::simulation::graph::DAGSimulation SofaSimulation;


namespace sofa::simplegui
{


typedef sofa::type::Vec3 Vec3;
typedef sofa::component::statecontainer::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;


SofaScene::SofaScene()
{
    _groot = _iroot = nullptr;
    std::shared_ptr<sofa::core::ObjectFactory::ClassEntry> classVisualModel;// = NULL;
	sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true, &classVisualModel);

    sofa::component::init();
}

void SofaScene::step( SReal dt)
{
    sofa::simulation::node::animate(_groot.get(),dt);
}

void SofaScene::printGraph()
{
    sofa::simulation::node::print(_groot.get());
}

void SofaScene::loadPlugins( const std::vector<std::string>& plugins )
{
    for (const auto& plugin : plugins)
    {
        cout <<"SofaScene::init, loading plugin " << plugin << '\n';
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugin);
    }

    sofa::helper::system::PluginManager::getInstance().init();
}

void SofaScene::open(const std::string& fileName )
{
    // --- Create simulation graph ---
    assert( !fileName.empty());

    if(_groot) sofa::simulation::node::unload (_groot);
    _groot = sofa::simulation::node::load( fileName.c_str() ).get();
    if(!_groot)
    {
        cerr << "loading failed" << endl;
        return;
    }

    _iroot = _groot->createChild("iroot").get();

//    _currentFileName = fileName;

    sofa::simulation::node::init(_groot.get());

    printGraph();
    SReal xm,xM,ym,yM,zm,zM;
    getBoundingBox(&xm,&xM,&ym,&yM,&zm,&zM);
    cout<<"SofaScene::setScene, xm="<<xm<<", xM"<< xM<< ", ym="<< ym<<", yM="<< yM<<", zm="<< zm<<", zM="<< zM<<endl;

}

void SofaScene::setScene(simulation::Node *node )
{
    if(_groot) sofa::simulation::node::unload (_groot);
    _groot = sofa::simulation::getSimulation()->createNewGraph("root").get();
    _groot->addChild(node);
    _iroot = _groot->createChild("iroot").get();
    sofa::simulation::node::init(_groot.get());
}

void SofaScene::reset()
{
    sofa::simulation::node::reset(_groot.get());
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

//    sofa::simulation::node::::init(_groot);
////    cout<<"SofaScene::init, scene loaded" << endl;
////    printGraph();
//}

void SofaScene::getBoundingBox( SReal* xmin, SReal* xmax, SReal* ymin, SReal* ymax, SReal* zmin, SReal* zmax )
{
    SReal pmin[3], pmax[3];
    sofa::simulation::node::computeBBox( _groot.get(), pmin, pmax );
    *xmin = pmin[0]; *xmax = pmax[0];
    *ymin = pmin[1]; *ymax = pmax[1];
    *zmin = pmin[2]; *zmax = pmax[2];
}

void SofaScene::insertInteractor( Interactor * interactor )
{
	if(_iroot)
	    _iroot->addChild(interactor->getNode());
}

simulation::Node* SofaScene::groot() { return _groot.get(); }

void SofaScene::initVisual(){
    sofa::simulation::node::initTextures(_groot.get());
}

void SofaScene::updateVisual()
{
    sofa::simulation::node::updateVisual(_groot.get()); // needed to update normals and VBOs ! (i think it should be better if updateVisual() was called from draw(), why it is not already the case ?)
}

//void SofaScene::draw( sofa::core::visual::VisualParams* v)
//{
//    if( v==NULL )
//    {
//
//    }
//    sofa::simulation::node::draw(_vparams, groot() );
//}


}
