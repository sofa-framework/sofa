#ifndef SOFA_HELPER_IO_MASSSPRINGLOADER_H
#define SOFA_HELPER_IO_MASSSPRINGLOADER_H

namespace sofa
{

namespace helper
{

namespace io
{

class MassSpringLoader
{
public:
    virtual ~MassSpringLoader() {}
    bool load(const char *filename);
    virtual void setNumMasses(int /*n*/) {}
    virtual void setNumSprings(int /*n*/) {}
    virtual void addMass(double /*px*/, double /*py*/, double /*pz*/, double /*vx*/, double /*vy*/, double /*vz*/, double /*mass*/, double /*elastic*/, bool /*fixed*/, bool /*surface*/) {}
    virtual void addSpring(int /*m1*/, int /*m2*/, double /*ks*/, double /*kd*/, double /*initpos*/) {}
    virtual void setGravity(double /*gx*/, double /*gy*/, double /*gz*/) {}
    virtual void setViscosity(double /*visc*/) {}
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
