#ifndef SOFA_SIMPLEGUI_VisualPickVisitor_H
#define SOFA_SIMPLEGUI_VisualPickVisitor_H
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/helper/vector.h>

namespace sofa
{
namespace simplegui
{

/**
 * @brief The VisualPickVisitor class displays the scene for OpenGL picking.
 * It uses the glPushName instruction, and it must be called within a specific OpenGL context, see http://www.lighthouse3d.com/opengl/picking/
 * The names vector member contains the names of all the objects rendered, in the traversal order.
 * This allows to associate a hit number to an object during the pre-processing.
 * @sa SofaGL
 * @author Francois Faure, 2014
 *
 * @warning The code is directly adapted from VisualDrawVisitor, without much insight
 */
class VisualPickVisitor : public  ::sofa::simulation::VisualVisitor
{
public:
    bool hasShader;
    VisualPickVisitor(core::visual::VisualParams* params);
    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);
    virtual void fwdVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    virtual void processVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    virtual void processObject(simulation::Node* node, core::objectmodel::BaseObject* o);
    virtual void bwdVisualModel(simulation::Node* node, core::visual::VisualModel* vm);
    virtual const char* getClassName() const { return "VisualPickVisitor"; }

    sofa::helper::vector<std::string> names; // names of the object displayed

private:
    int pickedId;


};

}}

#endif // SOFA_SIMPLEGUI_VisualPickVisitor_H
