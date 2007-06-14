#include "BatchGUI.h"
#include <sofa/simulation/tree/Simulation.h>

namespace sofa
{

namespace gui
{

BatchGUI::BatchGUI()
    : groot(NULL), nbIter(1000)
{
}

BatchGUI::~BatchGUI()
{
}

int BatchGUI::mainLoop()
{
    if (groot)
    {
        std::cout << "Computing "<<nbIter<<" iterations." << std::endl;
        for (int i=0; i<nbIter; i++)
        {
            sofa::simulation::tree::Simulation::animate(groot);
        }
        std::cout <<nbIter<<" iterations done." << std::endl;
    }
    return 0;
}

void BatchGUI::redraw()
{
}

int BatchGUI::closeGUI()
{
    delete this;
    return 0;
}

void BatchGUI::setScene(sofa::simulation::tree::GNode* groot, const char* filename)
{
    this->groot = groot;
    this->filename = (filename?filename:"");
}

sofa::simulation::tree::GNode* BatchGUI::currentSimulation()
{
    return groot;
}

SOFA_DECL_CLASS(BatchGUI)

int BatchGUIClass = SofaGUI::RegisterGUI("batch", &BatchGUI::CreateGUI, &BatchGUI::InitGUI, -1);
int BatchGUI::InitGUI(const char* /*name*/, const std::vector<std::string>& /*options*/)
{
    return 0;
}

SofaGUI* BatchGUI::CreateGUI(const char* /*name*/, const std::vector<std::string>& /*options*/, sofa::simulation::tree::GNode* groot, const char* filename)
{
    BatchGUI* gui = new BatchGUI();
    gui->setScene(groot, filename);
    return gui;
}

} // namespace gui

} // namespace sofa
