#ifndef SOFA_QTOGREVIEWER_PROPAGATEOGRESCENEMANAGER_H
#define SOFA_QTOGREVIEWER_PROPAGATEOGRESCENEMANAGER_H

#include <sofa/simulation/common/Visitor.h>
#include <Ogre.h>


namespace sofa
{
namespace core
{
namespace objectmodel
{
class BaseObject;
}
}
namespace simulation
{
class Node;
namespace ogre
{
class PropagateOgreSceneManager : public Visitor
{

public:
    PropagateOgreSceneManager(const core::ExecParams* params,Ogre::SceneManager* sceneMgr);
    Visitor::Result processNodeTopDown(simulation::Node* );


    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const { return "ogre"; }
    const char* getClassName() const { return "OgreInitVisitor"; }
protected:
    void processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* );

    Ogre::SceneManager*  mSceneMgr;
};
} // ogre
} // simulation
} // sofa
#endif //SOFA_QTOGREVIEWER_PROPAGATEOGRESCENEMANAGER_H
