#include <sofa/component/mass/DiagonalMass.inl>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

template <>
double DiagonalMass<RigidTypes, RigidMass>::getPotentialEnergy( const VecCoord& x )
{
    double e = 0;
    const MassVector &masses= f_mass.getValue().getArray();
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += theGravity.getVCenter()*masses[i].mass*x[i].getCenter();
    }
    return e;
}
template <>
void DiagonalMass<RigidTypes, RigidMass>::init()
{
    ForceField<RigidTypes>::init();
}
template <>
void DiagonalMass<RigidTypes, RigidMass>::handleEvent( Event * )
{
}

template <>
void DiagonalMass<RigidTypes, RigidMass>::draw()
{
    const MassVector &masses= f_mass.getValue().getArray();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mstate->getX();
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

template class DiagonalMass<RigidTypes,RigidMass>;

namespace helper   // \todo Why this must be inside helper namespace
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
void create(DiagonalMass<DataTypes, MassType>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< DiagonalMass<DataTypes, MassType>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
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

Creator<simulation::tree::xml::ObjectFactory, DiagonalMass<Vec3dTypes,double> > DiagonalMass3dClass("DiagonalMass",true);
Creator<simulation::tree::xml::ObjectFactory, DiagonalMass<Vec3fTypes,float > > DiagonalMass3fClass("DiagonalMass",true);
Creator<simulation::tree::xml::ObjectFactory, DiagonalMass<RigidTypes,RigidMass> > DiagonalMassRigidClass("DiagonalMass",true);

} // namespace mass

} // namespace component

} // namespace sofa

