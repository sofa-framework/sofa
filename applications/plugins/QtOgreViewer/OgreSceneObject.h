#ifndef SOFA_QTOGREVIEWER_OGRESCENEOBJECT_H
#define SOFA_QTOGREVIEWER_OGRESCENEOBJECT_H

#include <Ogre.h>

namespace sofa
{
namespace core
{
namespace ogre
{

class OgreSceneObject
{
public:
    void setSceneManager(Ogre::SceneManager* sceneMgr);
protected:
    Ogre::SceneManager* mSceneMgr;

};
}
}
}
#endif //SOFA_QTOGREVIEWER_OGRESCENEOBJECT_H
