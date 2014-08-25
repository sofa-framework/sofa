#include "SofaGLScene.h"

namespace sofa {
namespace newgui {



SofaGLScene::SofaGLScene(){
    sofaGL = new SofaGL(this);
}

SofaGLScene::~SofaGLScene() { delete sofaGL; }

void SofaGLScene::init( const std::string& fileName ){
    SofaScene::init(fileName);
    sofaGL->init();
}

void SofaGLScene::init( Node::SPtr groot ){
    SofaScene::init(groot);
    sofaGL->init();
}

void SofaGLScene::glDraw(){ sofaGL->draw(); }

void SofaGLScene::animate(){ SofaScene::step(0.04); }

PickedPoint SofaGLScene::pick( GLdouble ox, GLdouble oy, GLdouble oz, int x, int y ){ return sofaGL->pick(ox,oy,oz,x,y); }

void SofaGLScene::attach( Interactor*  i ) { sofaGL->attach(i); }

void SofaGLScene::move( Interactor* i, int x, int y){ sofaGL->move(i,x,y); }

void SofaGLScene::detach(Interactor* i){ sofaGL->detach(i); }




}// newgui
}// sofa

