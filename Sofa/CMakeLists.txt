cmake_minimum_required(VERSION 3.12)

### framework
# add Config only if it was not loaded before
if(NOT TARGET Sofa.Config)
    sofa_add_module(framework/Config Sofa.Config)
endif()

sofa_add_module(framework/Type Sofa.Type)
sofa_add_module(framework/Geometry Sofa.Geometry)
sofa_add_module(framework/Topology Sofa.Topology)
sofa_add_module(framework/Helper Sofa.Helper)
sofa_add_module(framework/LinearAlgebra Sofa.LinearAlgebra)
sofa_add_module(framework/DefaultType Sofa.DefaultType)
sofa_add_module(framework/Core Sofa.Core)
sofa_add_module(framework/Simulation Sofa.Simulation)
sofa_add_module(framework/Testing Sofa.Testing)

# Meta-module gathering all framework modules
sofa_add_module(framework/Framework Sofa.Framework)


### Component
sofa_add_module(Component Sofa.Component)
