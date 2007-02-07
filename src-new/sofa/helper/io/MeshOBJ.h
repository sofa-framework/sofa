#ifndef SOFA_HELPER_IO_MESHOBJ_H
#define SOFA_HELPER_IO_MESHOBJ_H

#include <sofa/helper/io/Mesh.h>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;

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

} // namespace io

} // namespace helper

} // namespace sofa

#endif
