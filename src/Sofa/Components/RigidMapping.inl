#ifndef SOFA_COMPONENTS_RIGIDMAPPING_INL
#define SOFA_COMPONENTS_RIGIDMAPPING_INL

#include "RigidMapping.h"
#include "MassSpringLoader.h"
#include "SphereLoader.h"
#include "Common/Mesh.h"
#include "GL/template.h"

#include "Sofa/Core/MechanicalMapping.inl"

#include <string>

namespace Sofa
{

namespace Components
{

using namespace Common;

template <class BaseMapping>
class RigidMapping<BaseMapping>::Loader : public MassSpringLoader, public SphereLoader
{
public:
    RigidMapping<BaseMapping>* dest;
    Loader(RigidMapping<BaseMapping>* dest) : dest(dest) {}
    virtual void addMass(double px, double py, double pz, double, double, double, double, double, bool, bool)
    {
        dest->points.push_back(Coord((Real)px,(Real)py,(Real)pz));
    }
    virtual void addSphere(double px, double py, double pz, double)
    {
        dest->points.push_back(Coord((Real)px,(Real)py,(Real)pz));
    }
};

template <class BaseMapping>
void RigidMapping<BaseMapping>::init(const char *filename)
{
    points.resize(0);

    if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".xs3"))
    {
        Loader loader(this);
        loader.MassSpringLoader::load(filename);
    }
    else if (strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".sph"))
    {
        Loader loader(this);
        loader.SphereLoader::load(filename);
    }
    else if (strlen(filename)>0)
    {
        // Default to mesh loader
        Mesh* mesh = Mesh::Create(filename);
        if (mesh!=NULL)
        {
            points.resize(mesh->getVertices().size());
            for (unsigned int i=0; i<mesh->getVertices().size(); i++)
                points[i] = (Coord)mesh->getVertices()[i];
            delete mesh;
        }
    }
}

template <class BaseMapping>
void RigidMapping<BaseMapping>::init()
{
    if (this->points.empty() && this->toModel!=NULL)
    {
        VecCoord& x = *this->toModel->getX();
        std::cout << "RigidMapping: init "<<x.size()<<" points."<<std::endl;
        points.resize(x.size());
        for (unsigned int i=0; i<x.size(); i++)
            points[i] = x[i];
    }
    this->BaseMapping::init();
}

template <class BaseMapping>
void RigidMapping<BaseMapping>::apply( typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    translation[0] = (Real)in[0].getCenter()[0];
    translation[1] = (Real)in[0].getCenter()[1];
    translation[2] = (Real)in[0].getCenter()[2];
    Real* q = orientation;
    q[0] = (Real)in[0].getOrientation()[0];
    q[1] = (Real)in[0].getOrientation()[1];
    q[2] = (Real)in[0].getOrientation()[2];
    q[3] = (Real)in[0].getOrientation()[3];
    rotation[0][0] =  (1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]));
    rotation[0][1] =  (2.0f * (q[0] * q[1] - q[2] * q[3]));
    rotation[0][2] =  (2.0f * (q[2] * q[0] + q[1] * q[3]));
    rotation[1][0] =  (2.0f * (q[0] * q[1] + q[2] * q[3]));
    rotation[1][1] =  (1.0f - 2.0f * (q[2] * q[2] + q[0] * q[0]));
    rotation[1][2] =  (2.0f * (q[1] * q[2] - q[0] * q[3]));
    rotation[2][0] =  (2.0f * (q[2] * q[0] - q[1] * q[3]));
    rotation[2][1] =  (2.0f * (q[1] * q[2] + q[0] * q[3]));
    rotation[2][2] =  (1.0f - 2.0f * (q[1] * q[1] + q[0] * q[0]));

    rotatedPoints.resize(points.size());
    out.resize(points.size());
    for(unsigned int i=0; i<points.size(); i++)
    {
        rotatedPoints[i] = rotation*points[i];
        out[i] = rotatedPoints[i];
        out[i] += translation;
    }
}

template <class BaseMapping>
void RigidMapping<BaseMapping>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
{
    Deriv v,omega;
    v[0] = (Real)in[0].getVCenter()[0];
    v[1] = (Real)in[0].getVCenter()[1];
    v[2] = (Real)in[0].getVCenter()[2];
    omega[0] = (Real)in[0].getVOrientation()[0];
    omega[1] = (Real)in[0].getVOrientation()[1];
    omega[2] = (Real)in[0].getVOrientation()[2];
    out.resize(points.size());
    for(unsigned int i=0; i<points.size(); i++)
    {
        // out = J in
        // J = [ I -OM^ ]
        out[i] =  v - cross(rotatedPoints[i],omega);
    }
}

template <class BaseMapping>
void RigidMapping<BaseMapping>::applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in )
{
    Deriv v,omega;
    v[0] = 0.0f;
    v[1] = 0.0f;
    v[2] = 0.0f;
    omega[0] = 0.0f;
    omega[1] = 0.0f;
    omega[2] = 0.0f;
    for(unsigned int i=0; i<points.size(); i++)
    {
        // out = Jt in
        // Jt = [ I     ]
        //      [ -OM^t ]
        // -OM^t = OM^

        Deriv f = in[i];
        v += f;
        omega += cross(rotatedPoints[i],f);
    }
    out[0].getVCenter()[0] += v[0];
    out[0].getVCenter()[1] += v[1];
    out[0].getVCenter()[2] += v[2];
    out[0].getVOrientation()[0] += omega[0];
    out[0].getVOrientation()[1] += omega[1];
    out[0].getVOrientation()[2] += omega[2];
}

template <class BaseMapping>
void RigidMapping<BaseMapping>::draw()
{
    if (!getShow(this)) return;
    glDisable (GL_LIGHTING);
    glPointSize(7);
    glColor4f (1,1,0,1);
    glBegin (GL_POINTS);
    typename Out::VecCoord& x = *this->toModel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        GL::glVertexT(x[i]);
    }
    glEnd();
}

} // namespace Components

} // namespace Sofa

#endif
