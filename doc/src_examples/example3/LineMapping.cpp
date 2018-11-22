#include "MassObject1d.h"

#include "Sofa/Core/MechanicalMapping.inl"
#include "Sofa/Components/XML/MappingNode.h"

using namespace Sofa::Components;
using namespace Sofa::Components::Common;
using namespace Sofa::Core;

class LineMapping : public Sofa::Core::MechanicalMapping< Sofa::Core::MechanicalObject<Vec1dTypes>, Sofa::Core::MechanicalObject<Vec3dTypes> >
{
public:
    typedef Sofa::Core::MechanicalMapping< Sofa::Core::MechanicalObject<Vec1dTypes>, Sofa::Core::MechanicalObject<Vec3dTypes> > BaseMapping;
    typedef BaseMapping::In In;
    typedef BaseMapping::Out Out;
    typedef Out::VecCoord VecCoord;
    typedef Out::VecDeriv VecDeriv;
    typedef Out::Coord Coord;
    typedef Out::Deriv Deriv;

    Coord p0;
    Deriv dx;

    LineMapping(In* from, Out* to)
        : BaseMapping(from, to), p0(0,0,0), dx(1,0,0)
    {
    }

    void apply( Out::VecCoord& out, const In::VecCoord& in )
    {
        out.resize(in.size());
        for(unsigned int i=0; i<out.size(); i++)
            out[i] = p0+dx*in[i];
    }

    void applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
    {
        out.resize(in.size());
        for(unsigned int i=0; i<out.size(); i++)
            out[i] = dx*in[i];
    }

    void applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
    {
        for(unsigned int i=0; i<out.size(); i++)
            out[i] += dx*in[i];
    }
};


namespace Sofa
{
namespace Components
{
namespace Common
{

void create(LineMapping*& obj, XML::Node<Sofa::Core::BasicMapping>* arg)
{
    XML::createWith2Objects< LineMapping, LineMapping::In, LineMapping::Out>(obj, arg);
    if (obj!=NULL)
    {
        obj->p0[0] = atof(arg->getAttribute("x0","0"));
        obj->p0[1] = atof(arg->getAttribute("y0","0"));
        obj->p0[2] = atof(arg->getAttribute("z0","0"));
        obj->dx[0] = atof(arg->getAttribute("dx","1"));
        obj->dx[1] = atof(arg->getAttribute("dy","0"));
        obj->dx[2] = atof(arg->getAttribute("dz","0"));
    }
}
}
}
}


Creator< XML::MappingNode::Factory, LineMapping > LineMappingClass("LineMapping", true);
