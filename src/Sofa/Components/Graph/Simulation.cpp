#include "Simulation.h"
#include "../XML/XML.h"
#include "../Common/SetDirectory.h"
#include "../init.h"
#include "PrintAction.h"
#include "AnimateAction.h"
#include "MechanicalAction.h"
#include "CollisionAction.h"
#include "UpdateMappingAction.h"
#include "ResetAction.h"
#include "VisualAction.h"

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

/// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void Simulation::animate(GNode* root, double dt)
{
    if (!root) return;
    AnimateAction act;
    act.setDt(dt);
    root->execute(act);
    root->execute<UpdateMappingAction>();
    root->execute<VisualUpdateAction>();
    root->execute<CollisionAction>();
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

/// Render the scene
void Simulation::draw(GNode* root)
{
    if (!root) return;
    root->execute<VisualDrawAction>();
}

} // namespace Graph

} // namespace Components

} // namespace Sofa

