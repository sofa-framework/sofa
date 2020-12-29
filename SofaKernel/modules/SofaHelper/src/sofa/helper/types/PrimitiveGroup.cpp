#include <sofa/helper/types/PrimitiveGroup.h>

namespace sofa::helper::types
{

std::ostream& operator << (std::ostream& out, const PrimitiveGroup &g)
{
    out << g.groupName << " " << g.materialName << " " << g.materialId << " " << g.p0 << " " << g.nbp;
    return out;
}

std::istream& operator >> (std::istream& in, PrimitiveGroup &g)
{
    in >> g.groupName >> g.materialName >> g.materialId >> g.p0 >> g.nbp;
    return in;
}

bool PrimitiveGroup::operator <(const PrimitiveGroup& p) const
{
    return p0 < p.p0;
}

PrimitiveGroup::PrimitiveGroup() : p0(0), nbp(0), materialId(-1) {}

PrimitiveGroup::PrimitiveGroup(int p0, int nbp, std::string materialName, std::string groupName, int materialId) : p0(p0), nbp(nbp), materialName(materialName), groupName(groupName), materialId(materialId) {}

void from_json(const sofa::helper::json& j, PrimitiveGroup& p)
{

}


} /// namespace sofa::helper::types
