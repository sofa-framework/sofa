/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/tree/ExportDotVisitor.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Colors.h>

#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/core/collision/ContactManager.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

ExportDotVisitor::ExportDotVisitor(const sofa::core::ExecParams* params /* PARAMS FIRST */, std::ostream* out)
    : GNodeVisitor(params),
      out(out),
      showNode(true),
      showObject(true),
      showBehaviorModel(true),
      showCollisionModel(true),
      showVisualModel(true),
      showMapping(true),
      showContext(true),
      showCollisionPipeline(true),
      showSolver(true),
      showMechanicalState(true),
      showForceField(true),
      showInteractionForceField(true),
      showConstraint(true),
      showMass(true),
      showTopology(true),
      showMechanicalMapping(true),
      labelNodeName(true),
      labelNodeClass(false),
      labelObjectName(true),
      labelObjectClass(true)
{
    *out << "digraph G {" << std::endl;
}

ExportDotVisitor::~ExportDotVisitor()
{
    *out << "}" << std::endl;
}

/// Test if a node should be displayed
bool ExportDotVisitor::display(GNode* node, const char **color)
{
    using namespace Colors;
    if (!node) return false;
    if (showNode)
    {
        if (color) *color = COLOR[NODE];
        return true;
    }
    else
        return false;
}

/// Test if an object should be displayed
bool ExportDotVisitor::display(core::objectmodel::BaseObject* obj, const char **color)
{
    using namespace Colors;
    const char* c = NULL;
    if (color==NULL) color=&c;
    if (!obj) return false;
    if (!showObject) return false;
    *color = COLOR[OBJECT];
    bool show = false;
    bool hide = false;
    if (dynamic_cast<core::behavior::BaseMechanicalState*>(obj))
    {
        if (showMechanicalState) { show = true; *color = COLOR[MMODEL]; }
        else hide = true;
    }
    if (dynamic_cast<core::behavior::BaseMass*>(obj))
    {
        if (showMass) { show = true; *color = COLOR[MASS]; }
        else hide = true;
    }
    if (dynamic_cast<core::topology::Topology *>(obj))
    {
        if (showTopology) { show = true; *color = COLOR[TOPOLOGY]; }
        else hide = true;
    }
    if (dynamic_cast<core::CollisionModel*>(obj))
    {
        if (showCollisionModel) { show = true; *color = COLOR[CMODEL]; }
        else hide = true;
    }
    if (dynamic_cast<core::BaseMapping*>(obj))
    {
        if (showMapping) { show = true; *color = COLOR[MAPPING]; }
        else hide = true;
    }
    if (dynamic_cast<core::objectmodel::ContextObject*>(obj))
    {
        if (showContext) { show = true; *color = COLOR[CONTEXT]; }
        else hide = true;
    }
    if (dynamic_cast<core::collision::Pipeline*>(obj)
        || dynamic_cast<core::collision::Intersection*>(obj)
        || dynamic_cast<core::collision::Detection*>(obj)
        || dynamic_cast<core::collision::ContactManager*>(obj)
        || dynamic_cast<core::collision::CollisionGroupManager*>(obj))
    {
        if (showCollisionPipeline) { show = true; *color = COLOR[COLLISION]; }
        else hide = true;
    }
    if (dynamic_cast<core::behavior::OdeSolver*>(obj))
    {
        if (showSolver) { show = true; *color = COLOR[SOLVER]; }
        else hide = true;
    }
    if (dynamic_cast<core::behavior::BaseInteractionForceField*>(obj) &&
        dynamic_cast<core::behavior::BaseInteractionForceField*>(obj)->getMechModel1()!=dynamic_cast<core::behavior::BaseInteractionForceField*>(obj)->getMechModel2())
    {
        if (showInteractionForceField) { show = true; *color = COLOR[IFFIELD]; }
        else hide = true;
    }
    else if (dynamic_cast<core::behavior::BaseForceField*>(obj))
    {
        if (showForceField) { show = true; *color = COLOR[FFIELD]; }
        else hide = true;
    }
    if (dynamic_cast<core::behavior::BaseProjectiveConstraintSet*>(obj))
    {
        if (showConstraint) { show = true; *color = COLOR[PROJECTIVECONSTRAINTSET]; }
        else hide = true;
    }
    if (dynamic_cast<core::behavior::BaseConstraintSet*>(obj))
    {
        if (showConstraint) { show = true; *color = COLOR[CONSTRAINTSET]; }
        else hide = true;
    }
    if (dynamic_cast<core::BehaviorModel*>(obj))
    {
        if (showBehaviorModel) { show = true; *color = COLOR[BMODEL]; }
        else hide = true;
    }

    if (dynamic_cast<core::visual::VisualModel*>(obj) && !hide && !show)
    {
        if (showVisualModel) { show = true; *color = COLOR[VMODEL]; }
        else hide = true;
    }

    return show || !hide;
}

/// Find the node or object a given object should be attached to.
/// This is the parent node if it is displayed, otherwise it is the attached MechanicalState or Solver.
/// Returns an empty string if not found.
std::string ExportDotVisitor::getParentName(core::objectmodel::BaseObject* obj)
{
    GNode* node = dynamic_cast<GNode*>(obj->getContext());
    if (!node) return "";
    if (display(node))
        return getName(node);
    if (dynamic_cast<core::BaseMapping*>(obj))
        return "";
    if (!node->collisionPipeline.empty() && display(node->collisionPipeline) &&
        (dynamic_cast<core::collision::Intersection*>(obj) ||
                dynamic_cast<core::collision::Detection*>(obj) ||
                dynamic_cast<core::collision::ContactManager*>(obj) ||
                dynamic_cast<core::collision::CollisionGroupManager*>(obj)))
        return getName(node->collisionPipeline);
    /// \todo consider all solvers instead of the first one (FF)
    if (node->mechanicalState!=obj && node->solver[0]!=obj  && node->animationManager!=obj && display(node->mechanicalState))
        return getName(node->mechanicalState);
    if (node->solver[0]!=obj && node->animationManager!=obj && display(node->solver[0]))
        return getName(node->solver[0]);
    if (node->animationManager!=obj && display(node->solver[0]))
        return getName(node->animationManager);
    if ((node->mechanicalState==obj || node->solver[0]==obj) && !node->mechanicalMapping && node->parent() && display(node->parent()->solver[0]))
        return getName(node->parent()->solver[0]);
    if ((node->mechanicalState==obj || node->solver[0]==obj || node->animationManager==obj) && !node->mechanicalMapping && node->parent() && display(node->parent()->animationManager))
        return getName(node->parent()->animationManager);
    return "";
}

/// Compute the name of a given node or object
std::string ExportDotVisitor::getName(core::objectmodel::Base* o, std::string prefix)
{
    if (names.count(o)>0)
        return names[o];
    std::string oname = o->getName();
    std::string name = prefix;
    for (unsigned i = 0; i<oname.length(); i++)
    {
        char c = oname[i];
        static const char *chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        if (strchr(chars, c))
            name += c;
    }
    if (name.length() > prefix.length())
        name += '_';
    int index = nextIndex[name]++;
    if (index)
    {
        char str[16]={"azertyazertyaze"};
        snprintf(str,sizeof(str),"%d",index+1);
        name += str;
    }
    names[o] = name;
    return name;
}

/// Compute the name of a given node
std::string ExportDotVisitor::getName(GNode* node)
{
    return getName(node, "n_");
}

/// Compute the name of a given object
std::string ExportDotVisitor::getName(core::objectmodel::BaseObject* obj)
{
    return getName(obj, "o_");
}

void ExportDotVisitor::processObject(GNode* /*node*/, core::objectmodel::BaseObject* obj)
{
    //std::cout << ' ' << obj->getName() << '(' << sofa::helper::gettypename(typeid(*obj)) << ')';
    const char* color=NULL;
    if (display(obj,&color))
    {
        std::string name = getName(obj);
        *out << name << " [shape=box,";
        if (color!=NULL)
            *out << "style=\"filled\",fillcolor=\"" << color << "\",";
        *out << "label=\"";
        if (labelObjectClass)
        {
            std::string name = helper::gettypename(typeid(*obj));
            std::string::size_type pos = name.find('<');
            if (pos != std::string::npos)
                name.erase(pos);
            *out << name;
            if (labelObjectName)
                *out << "\\n";
        }
        if (labelObjectName)
        {
            if (std::string(obj->getName(),0,7) != "default")
                *out << obj->getName();
        }
        *out << "\"];" << std::endl;
        std::string pname = getParentName(obj);
        if (!pname.empty())
        {
            *out << pname << " -> " << name;
            if (dynamic_cast<core::BaseMapping*>(obj))
                *out << "[constraint=false]";
            *out << ";" << std::endl;
        }
        core::behavior::BaseInteractionForceField* iff = dynamic_cast<core::behavior::BaseInteractionForceField*>(obj);
        if (iff!=NULL)
        {
            core::behavior::BaseMechanicalState* model1 = iff->getMechModel1();
            core::behavior::BaseMechanicalState* model2 = iff->getMechModel2();
            if (model1 != model2)
            {
                if (display(model1))
                    *out << name << " -> " << getName(model1) << " [style=\"dashed\",arrowhead=\"open\"];" << std::endl;
                if (display(model2))
                    *out << name << " -> " << getName(model2) << " [style=\"dashed\",arrowhead=\"open\"];" << std::endl;
            }
        }
        core::BaseMapping* map = dynamic_cast<core::BaseMapping*>(obj);
        if (map!=NULL)
        {
            core::objectmodel::BaseObject* model1 = map->getFrom()[0];
            core::objectmodel::BaseObject* model2 = map->getTo()[0];
            if (display(model1))
            {
                *out << getName(model1) << " -> " << name << " [style=\"dashed\",arrowhead=\"none\"";
                core::BaseMapping* bmm = dynamic_cast<core::BaseMapping*>(obj);
                if (bmm)
                {
                    if(bmm->isMechanical())
                        *out << ",arrowtail=\"open\"";
                }
                *out << "];" << std::endl;
            }
            if (display(model2))
                *out << name << " -> " << getName(model2) << " [style=\"dashed\"];" << std::endl;
        }
    }
}

simulation::Visitor::Result ExportDotVisitor::processNodeTopDown(GNode* node)
{
    const char* color=NULL;
    if (display(node,&color))
    {
        *out << getName(node) << " [shape=hexagon,width=0.25,height=0.25,style=\"filled\"";
        if (color) *out << ",fillcolor=\"" << color << "\"";
        *out << ",label=\"";
        if (labelNodeClass)
        {
            std::string name = helper::gettypename(typeid(*node));
            std::string::size_type pos = name.find('<');
            if (pos != std::string::npos)
                name.erase(pos);
            *out << name;
            if (labelNodeName)
                *out << "\\n";
        }
        if (labelNodeName)
        {
            if (std::string(node->getName(),0,7) != "default")
                *out << node->getName();
        }
        *out << "\"];" << std::endl;
        if (node->parent())
        {
            *out << getName(node->parent()) << " -> " << getName(node)<< " [minlen=2,style=\"bold\"];" << std::endl;
        }
    }

    for (GNode::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        this->processObject(node, it->get());
    }

    return RESULT_CONTINUE;
}

void ExportDotVisitor::processNodeBottomUp(GNode* /*node*/)
{
}

} // namespace tree

} // namespace simulation

} // namespace sofa

