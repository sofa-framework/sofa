#ifndef SOFA_COMPONENTS_SPHERELOADER_H
#define SOFA_COMPONENTS_SPHERELOADER_H

namespace Sofa
{

namespace Components
{

class SphereLoader
{
public:
    virtual ~SphereLoader() {}
    bool load(const char *filename);
    virtual void setNumSpheres(int /*n*/) {}
    virtual void addSphere(double /*px*/, double /*py*/, double /*pz*/, double /*r*/) {}
};

} // namespace Components

} // namespace Sofa

#endif
