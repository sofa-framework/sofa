#ifndef PYTHONMODULE_SOFA_BINDING_SIMULATION_H
#define PYTHONMODULE_SOFA_BINDING_SIMULATION_H

#include <SofaPython3/Sofa/Core/Binding_BaseObject.h>

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation;

template class pybind11::class_<Simulation, Simulation::SPtr>;

namespace sofapython3
{

void moduleAddSimulation(py::module &m);
void moduleAddRuntime(py::module &m);

} ///sofapython3

#endif /// PYTHONMODULE_SOFA_BINDING_SIMULATION_H
