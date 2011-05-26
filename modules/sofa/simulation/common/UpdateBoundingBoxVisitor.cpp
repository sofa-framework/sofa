#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{
namespace simulation
{

UpdateBoundingBoxVisitor::UpdateBoundingBoxVisitor(const sofa::core::ExecParams* params)
    :Visitor(params)
{

}

Visitor::Result UpdateBoundingBoxVisitor::processNodeTopDown(Node* node)
{
    using namespace sofa::core::objectmodel;
    helper::vector<BaseObject*> objectList;
    helper::vector<BaseObject*>::iterator object;
    node->get<BaseObject>(&objectList,BaseContext::Local);
    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    nodeBBox->invalidate();
    for ( object = objectList.begin(); object != objectList.end(); ++object)
    {
        (*object)->computeBBox(params);
        nodeBBox->include((*object)->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);
    return RESULT_CONTINUE;
}

void UpdateBoundingBoxVisitor::processNodeBottomUp(simulation::Node* node)
{
    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    Node::ChildIterator childNode;
    for( childNode = node->child.begin(); childNode!=node->child.end(); ++childNode)
    {
        nodeBBox->include((*childNode)->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);
}

}
}
