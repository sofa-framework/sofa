#include "PropagateOgreSceneManager.h"
#include "OgreSceneObject.h"
#include <sofa/simulation/common/Node.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{
namespace simulation
{
namespace ogre
{

PropagateOgreSceneManager::PropagateOgreSceneManager(const core::ExecParams* params, Ogre::SceneManager* sceneMgr)
    :Visitor(params),
     mSceneMgr(sceneMgr)
{

}

Visitor::Result PropagateOgreSceneManager::processNodeTopDown(simulation::Node* node)
{
    Visitor::for_each(this, node, node->object, &PropagateOgreSceneManager::processObject);
    return RESULT_CONTINUE;
}

void PropagateOgreSceneManager::processObject(simulation::Node* /*node*/ ,core::objectmodel::BaseObject* obj)
{
    core::ogre::OgreSceneObject* ogreObj;
    if( (ogreObj = dynamic_cast<core::ogre::OgreSceneObject*>(obj)) != NULL )
    {
        ogreObj->setSceneManager(mSceneMgr);
    }
}
}
}
}
