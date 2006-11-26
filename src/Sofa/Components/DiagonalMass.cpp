#include "DiagonalMass.inl"
#include "Sofa/Components/Common/ObjectFactory.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"
#include "GL/Axis.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <>
void DiagonalMass<RigidTypes, RigidMass>::draw()
{
    const VecMass& masses = f_mass.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mmodel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        const Quat& orient = x[i].getOrientation();
        //orient[3] = -orient[3];
        const RigidTypes::Vec3& center = x[i].getCenter();
        RigidTypes::Vec3 len;
        // The moment of inertia of a box is:
        //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
        //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
        //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
        // So to get lx,ly,lz back we need to do
        //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
        // Note that RigidMass inertiaMatrix is already divided by M
        double m00 = masses[i].inertiaMatrix[0][0];
        double m11 = masses[i].inertiaMatrix[1][1];
        double m22 = masses[i].inertiaMatrix[2][2];
        len[0] = sqrt(m11+m22-m00);
        len[1] = sqrt(m00+m22-m11);
        len[2] = sqrt(m00+m11-m22);

        GL::Axis::draw(center, orient, len);
    }
}

SOFA_DECL_CLASS(DiagonalMass)

template class DiagonalMass<Vec3dTypes,double>;
template class DiagonalMass<Vec3fTypes,float>;

// specialization for rigid bodies
template <>
double DiagonalMass<RigidTypes,RigidMass>::getPotentialEnergy( const RigidTypes::VecCoord& x )
{
    const VecMass& masses = f_mass.getValue();
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<masses.size(); i++)
    {
        e += g*masses[i].mass*x[i].getCenter();
    }
    return e;
}

template class DiagonalMass<RigidTypes,RigidMass>;

namespace Common   // \todo Why this must be inside Common namespace
{
template<class Vec>
void readVec1(Vec& vec, const char* str)
{
    vec.clear();
    if (str==NULL) return;
    const char* str2 = NULL;
    for(;;)
    {
        double v = strtod(str,(char**)&str2);
        if (str2==str) break;
        str = str2;
        vec.push_back((typename Vec::value_type)v);
    }
}

template<class DataTypes, class MassType>
void create(DiagonalMass<DataTypes, MassType>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< DiagonalMass<DataTypes, MassType>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("filename"))
        {
            obj->load(arg->getAttribute("filename"));
            arg->removeAttribute("filename");
        }
        obj->parseFields( arg->getAttributeMap() );

        /*
        if (arg->getAttribute("gravity"))
        {
        	double x=0;
        	double y=0;
        	double z=0;
        	sscanf(arg->getAttribute("gravity"),"%lf %lf %lf",&x,&y,&z);
        	typename DataTypes::Deriv g;
        	DataTypes::set(g,x,y,z);
        	obj->setGravity(g);
        }
        */
        if (arg->getAttribute("mass"))
        {
            std::vector<MassType> mass;
            readVec1(mass,arg->getAttribute("mass"));
            obj->clear();
            for (unsigned int i=0; i<mass.size(); i++)
                obj->addMass(mass[i]);
        }
    }
}
}

Creator< ObjectFactory, DiagonalMass<Vec3dTypes,double> > DiagonalMass3dClass("DiagonalMass",true);
Creator< ObjectFactory, DiagonalMass<Vec3fTypes,float > > DiagonalMass3fClass("DiagonalMass",true);
Creator< ObjectFactory, DiagonalMass<RigidTypes,RigidMass> > DiagonalMassRigidClass("DiagonalMass",true);

} // namespace Components

} // namespace Sofa
