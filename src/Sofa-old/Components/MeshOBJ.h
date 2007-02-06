#ifndef SOFA_COMPONENTS_MESHOBJ_H
#define SOFA_COMPONENTS_MESHOBJ_H

#include "Common/Mesh.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class MeshOBJ : public Mesh
{
private:

    std::vector<Material> materials;

    void readOBJ (FILE *file);
    void readMTL (char *filename);

public:

    MeshOBJ(const std::string& filename)
    {
        init (filename);
    }

    void init (std::string filename);
};

} // namespace Components

} // namespace Sofa

#endif
