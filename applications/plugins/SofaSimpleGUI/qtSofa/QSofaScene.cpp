#include "QSofaScene.h"

namespace sofa {
namespace newgui {

QSofaScene::QSofaScene(QObject *parent) :
    QObject(parent)
  , _dt(0.04)
{
}

void QSofaScene::step()
{
    SofaScene::step(_dt);
    emit stepEnd(_dt);
}

void QSofaScene::setTimeStep( SReal dt ){
    _dt = dt;
}

SReal QSofaScene::dt() const { return _dt; }


}//newgui
}//sofa
