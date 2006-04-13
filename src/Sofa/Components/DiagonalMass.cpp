#include "DiagonalMass.inl"
#include "Sofa/Components/XML/MassNode.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"
#include "Scene.h"
#include "GL/Repere.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <>
void DiagonalMass<RigidTypes, RigidMass>::draw()
{
    if (!Scene::getInstance()->getShowBehaviorModels()) return;
    VecCoord& x = *mmodel->getX();
    for (unsigned int i=0; i<x.size(); i++)
    {
        Quat orient = x[i].getOrientation();
        RigidTypes::Vec3& center = x[i].getCenter();
        orient[3] = -orient[3];

        static GL::Axis *axis = new GL::Axis(center, orient);

        axis->update(center, orient);
        axis->draw();
    }
}

SOFA_DECL_CLASS(DiagonalMass)

template class DiagonalMass<Vec3dTypes,double>;
template class DiagonalMass<Vec3fTypes,float>;

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
        vec.push_back(v);
    }
}

template<class DataTypes, class MassType>
void create(DiagonalMass<DataTypes, MassType>*& obj, XML::Node<Core::Mass>* arg)
{
    XML::createWithParent< DiagonalMass<DataTypes, MassType>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("filename"))
        {
            obj->load(arg->getAttribute("filename"));
        }
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
        if (arg->getAttribute("mass"))
        {
            std::vector<double> mass;
            readVec1(mass,arg->getAttribute("mass"));
            obj->clear();
            for (unsigned int i=0; i<mass.size(); i++)
                obj->addMass(mass[i]);
        }
    }
}
}

Creator< XML::MassNode::Factory, DiagonalMass<Vec3dTypes,double> > DiagonalMass3dClass("DiagonalMass",true);
Creator< XML::MassNode::Factory, DiagonalMass<Vec3fTypes,float > > DiagonalMass3fClass("DiagonalMass",true);
Creator< XML::MassNode::Factory, DiagonalMass<RigidTypes,RigidMass> > DiagonalMassRigidClass("DiagonalMass",true);

} // namespace Components

} // namespace Sofa
