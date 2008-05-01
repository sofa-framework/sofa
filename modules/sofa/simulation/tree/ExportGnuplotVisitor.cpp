#include <sofa/simulation/tree/ExportGnuplotVisitor.h>
#include <sofa/component/System.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{

namespace tree
{

Visitor::Result InitGnuplotVisitor::processNodeTopDown(component::System* node)
{
    if (node->mechanicalState != NULL )
    {
        node->mechanicalState->initGnuplot(getSimulation()->gnuplotDirectory.getValue());
    }
    if (node->mass != NULL )
    {
        node->mass->initGnuplot(getSimulation()->gnuplotDirectory.getValue());
    }
    return RESULT_CONTINUE;
}

ExportGnuplotVisitor::ExportGnuplotVisitor( double time )
    : m_time(time)
{}

Visitor::Result ExportGnuplotVisitor::processNodeTopDown(component::System* node)
{
    if (node->mechanicalState != NULL )
    {
        node->mechanicalState->exportGnuplot(m_time);
    }
    if (node->mass!= NULL )
    {
        node->mass->exportGnuplot(m_time);
    }
    return RESULT_CONTINUE;
}


} // namespace tree

} // namespace simulation

} // namespace sofa

