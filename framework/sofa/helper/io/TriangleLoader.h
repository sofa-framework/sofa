#ifndef SOFA_HELPER_IO_TRIANGLELOADER_H
#define SOFA_HELPER_IO_TRIANGLELOADER_H

#include <stdio.h>

namespace sofa
{

namespace helper
{

namespace io
{

class TriangleLoader
{
public:
    virtual ~TriangleLoader() {}
    bool load(const char *filename);
    virtual void addVertices (double /*x*/, double /*y*/, double /*z*/) {};
    virtual void addTriangle (int /* idp1 */, int /*idp2*/, int /*idp3*/) {};

private:
    void loadTriangles(FILE *file);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
