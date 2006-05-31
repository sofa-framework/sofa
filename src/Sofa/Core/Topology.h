#ifndef SOFA_CORE_TOPOLOGY_H
#define SOFA_CORE_TOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class Topology : public virtual Abstract::BaseObject
{
public:
    virtual ~Topology() { }

    // Access to embedded position information (in case the topology is a regular grid for instance)
    // This is not very clean and is quit slow but it should only be used during initialization

    virtual bool hasPos() { return false; }
    virtual int getNbPoints() { return 0; }
    virtual double getPX(int /*i*/) { return 0.0; }
    virtual double getPY(int /*i*/) { return 0.0; }
    virtual double getPZ(int /*i*/) { return 0.0; }
};

} // namespace Core

} // namespace Sofa

#endif
