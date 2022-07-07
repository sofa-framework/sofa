cmake_minimum_required(VERSION 3.12)

### framework
# add Config only if it was not loaded before
if(NOT TARGET Sofa.Config)
    sofa_add_module(framework/Config Sofa.Config ON)
endif()

sofa_add_module(framework/Type Sofa.Type ON)
sofa_add_module(framework/Geometry Sofa.Geometry ON)
sofa_add_module(framework/Topology Sofa.Topology ON)
sofa_add_module(framework/Helper Sofa.Helper ON)
sofa_add_module(framework/LinearAlgebra Sofa.LinearAlgebra ON)
sofa_add_module(framework/DefaultType Sofa.DefaultType ON)
sofa_add_module(framework/Core Sofa.Core ON)
sofa_add_module(framework/Simulation Sofa.Simulation ON)
sofa_add_module(framework/Testing Sofa.Testing ON)

# Meta-module gathering all framework modules
sofa_add_module(framework/Framework Sofa.Framework ON)


### Component
sofa_add_module(Component Sofa.Component ON)
