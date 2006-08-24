#include "Simulation.h"
#include "../XML/XML.h"
#include "../Common/SetDirectory.h"
#include "../init.h"
#include "PrintAction.h"
#include "AnimateAction.h"
#include "MechanicalAction.h"
#include "CollisionAction.h"
#include "UpdateContextAction.h"
#include "UpdateMappingAction.h"
#include "ResetAction.h"
#include "VisualAction.h"
#include "DeleteAction.h"

namespace Sofa
{

namespace Components
{

namespace Graph
{

using namespace Common;

/// Load a scene from a file
GNode* Simulation::load(const char *filename)
{
    ::Sofa::Components::init();
    std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    XML::BaseNode* xml = XML::load(filename);
    if (xml==NULL)
    {
        return NULL;
    }

    // We go the the current file's directory so that all relative path are correct
    SetDirectory chdir(filename);

    std::cout << "Initializing objects"<<std::endl;
    if (!xml->init())
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    GNode* root = dynamic_cast<GNode*>(xml->getBaseObject());
    if (root == NULL)
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return NULL;
    }

    std::cout << "Initializing simulation "<<root->getName()<<std::endl;

    root->init();

    // As mappings might be initialized after visual models, it is necessary to update them
    root->execute<VisualUpdateAction>();

    std::cout << "load done."<<std::endl;

    delete xml;

    return root;
}

/// Print all object in the graph
void Simulation::print(GNode* root)
{
    if (!root) return;
    root->execute<PrintAction>();
}

/// Initialize the scene.
// void Simulation::init(GNode* root)
// {
// 	if (!root) return;
//         // There will probably be an InitAction on day...
// 	root->execute<MechanicalPropagatePositionAndVelocityAction>();
// }

/// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void Simulation::animate(GNode* root, double dt)
{
    double nextTime = root->getTime() + root->getDt();
    if (!root) return;
    //std::cout << "animate\n";

    AnimateAction act;
    act.setDt(dt);
    root->execute(act);

    root->execute<CollisionAction>();

    root->execute<UpdateMappingAction>();
    root->execute<VisualUpdateAction>();
    root->setTime( nextTime );
}

/// Reset to initial state
void Simulation::reset(GNode* root)
{
    if (!root) return;
    root->execute<ResetAction>();
    root->execute<MechanicalPropagatePositionAndVelocityAction>();
    root->execute<UpdateMappingAction>();
    root->execute<VisualUpdateAction>();
}

/// Initialize the textures
void Simulation::initTextures(GNode* root)
{
    if (!root) return;
    root->execute<VisualInitTexturesAction>();
}

/// Update contexts. Required before drawing the scene if root flags are modified.
void Simulation::updateContext(GNode* root)
{
    if (!root) return;
    root->execute<UpdateContextAction>();
}

/// Render the scene
void Simulation::draw(GNode* root)
{
    if (!root) return;
    //std::cout << "draw\n";
    root->execute<VisualDrawAction>();
}

/// Delete a scene from memory. After this call the pointer is invalid
void Simulation::unload(GNode* root)
{
    if (!root) return;
    root->execute<DeleteAction>();
    if (root->getParent()!=NULL)
        root->getParent()->removeChild(root);
    delete root;
}

} // namespace Graph

} // namespace Components

} // namespace Sofa

