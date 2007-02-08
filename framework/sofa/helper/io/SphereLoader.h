#ifndef SOFA_HELPER_IO_SPHERELOADER_H
#define SOFA_HELPER_IO_SPHERELOADER_H

namespace sofa
{

namespace helper
{

namespace io
{

class SphereLoader
{
public:
    virtual ~SphereLoader() {}
    bool load(const char *filename);
    virtual void setNumSpheres(int /*n*/) {}
    virtual void addSphere(double /*px*/, double /*py*/, double /*pz*/, double /*r*/) {}
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
