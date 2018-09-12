#ifndef PYTHONMODULE_SOFA_BINDING_SIMULATION_H
#define PYTHONMODULE_SOFA_BINDING_SIMULATION_H

#include "Binding_BaseObject.h"

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation;

template class py::class_<Simulation, Simulation::SPtr>;

void moduleAddSimulation(py::module &m);

#endif /// PYTHONMODULE_SOFA_BINDING_SIMULATION_H
