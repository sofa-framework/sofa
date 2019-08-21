#include "MassObject1d.h"
#include "Sofa/Components/MassObject.inl"
#include "Sofa/Components/XML/DynamicNode.h"

using namespace Sofa::Components;
using namespace Sofa::Components::Common;
using namespace Sofa::Core;

/// Read a vector of scalars from a string.
void readVec1(std::vector<double>& vec, const char* str)
{
    vec.clear();
    if (str==NULL) return;
    const char* str2 = NULL;
    for(;;)
    {
        double v = strtod(str,(char**)&str2);
        std::cout << v << std::endl;
        if (str2==str) break;
        str = str2;
        vec.push_back(v);
    }
}

namespace Sofa
{
namespace Components
{
namespace Common
{
/// Construct a MassObject1d object from a XML node.
void create(MassObject1d*& obj, XML::Node<Sofa::Core::DynamicModel>* arg)
{
    obj = new MassObject1d();
    obj->clear();
    std::vector<double> mass;
    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<double> fixed;
    readVec1(mass,arg->getAttribute("mass"));
    readVec1(pos,arg->getAttribute("position"));
    readVec1(vel,arg->getAttribute("velocity"));
    readVec1(fixed,arg->getAttribute("fixed"));
    if (arg->getAttribute("gravity"))
    {
        obj->setGravity(atof(arg->getAttribute("gravity")));
    }
    unsigned int maxsize = mass.size();
    if (pos.size()>maxsize) maxsize = pos.size();
    if (vel.size()>maxsize) maxsize = vel.size();
    double defaultmass = (mass.empty()?1.0:*mass.rbegin());
    while (mass.size()<maxsize)
        mass.push_back(defaultmass);
    double defaultpos = 0;
    if (!pos.empty()) defaultpos = *pos.rbegin();
    while (pos.size()<maxsize)
        pos.push_back(defaultpos);
    double defaultvel = 0;
    if (!vel.empty()) defaultvel = *vel.rbegin();
    while (vel.size()<maxsize)
        vel.push_back(defaultvel);
    for (unsigned int i=0; i<maxsize; i++)
    {
        obj->addMass(pos[i], vel[i], mass[i], 0.0, (std::find(fixed.begin(), fixed.end(), (double)i)!=fixed.end()));
    }
}
}
}
}

Creator< XML::DynamicNode::Factory, MassObject<Vec1dTypes> > MassObject1dClass("MassObject1d");
