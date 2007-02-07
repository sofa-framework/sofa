#ifndef SOFA_SIMULATION_TREE_COLORS_H
#define SOFA_SIMULATION_TREE_COLORS_H

namespace sofa
{

namespace simulation
{

namespace tree
{

namespace Colors
{

enum
{
    NODE = 0,
    OBJECT = 1,
    CONTEXT = 2,
    BMODEL = 3,
    CMODEL = 4,
    MMODEL = 5,
    CONSTRAINT = 6,
    IFFIELD = 7,
    FFIELD = 8,
    SOLVER = 9,
    COLLISION = 10,
    MMAPPING = 11,
    MAPPING = 12,
    MASS = 13,
    TOPOLOGY = 14,
    VMODEL = 15,
};

// See http://www.graphviz.org/doc/info/colors.html
// The following is mostly the "set312" colors

static const char* COLOR[16]=
{
    /*Node                  =*/ "#dedede", // color 9
    /*Object                =*/ "#ffffff", // white
    /*Context               =*/ "#d7191c", // color spectral4/1
    /*BehaviorModel         =*/ "#93ff49", // color 7 (brighter)
    /*CollisionModel        =*/ "#fccde5", // color 8
    /*MechanicalState       =*/ "#8dd3c7", // color 1
    /*Constraint            =*/ "#fdb462", // color 6
    /*InteractionForceField =*/ "#fb8072", // color 4
    /*ForceField            =*/ "#bebada", // color 3
    /*Solver                =*/ "#b3de69", // color 7
    /*CollisionPipeline     =*/ "#bc80bd", // color 10
    /*MechanicalMapping     =*/ "#2b83da", // color spectral4/4
    /*Mapping               =*/ "#80b1d3", // color 5
    /*Mass                  =*/ "#ffffb3", // color 2
    /*Topology              =*/ "#ffed6f", // color 12
    /*VisualModel           =*/ "#eefdea", // color 11 (brighter)
};

}

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
