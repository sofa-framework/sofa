#include "OscillatorConstraint.inl"
#include "Sofa/Components/Common/ObjectFactory.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"
//#include "GL/Axis.h"
#include <sstream>

namespace Sofa
{

namespace Components
{

using namespace Common;

//template <>
//void OscillatorConstraint<RigidTypes>::draw()
//{
///*	if (!getContext()->getShowBehaviorModels()) return;
//	VecCoord& x = *mmodel->getX();
//	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
//	{
//		int i = *it;
//		Quat orient = x[i].getOrientation();
//		RigidTypes::Vec3& center = x[i].getCenter();
//		orient[3] = -orient[3];
//
//		static GL::Axis *axis = new GL::Axis(center, orient, 5);
//
//		axis->update(center, orient);
//		axis->draw();
//	}*/
//	if (!getContext()->getShowBehaviorModels()) return;
//	VecCoord& x = *mmodel->getX();
//	glDisable (GL_LIGHTING);
//	glPointSize(10);
//	glColor4f (1,0.5,0.5,1);
//	glBegin (GL_POINTS);
//	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
//	{
//	   GL::glVertexT(x[0].getCenter());
//	}
//	glEnd();
//}
//
//template <>
//      void OscillatorConstraint<RigidTypes>::projectResponse(VecDeriv& res)
//{
//   res[0] = Deriv();
//}


SOFA_DECL_CLASS(OscillatorConstraint)

template class OscillatorConstraint<Vec3dTypes>;
template class OscillatorConstraint<Vec3fTypes>;

namespace Common   // \todo Why this must be inside Common namespace
{

/** Clear the container and fill it with values read from the string */
template<class C>
void readVector( C& container, const char* string )
{
    typedef typename C::value_type value_type;
    value_type v;
    container.clear();
    std::istringstream in(string);
    while( in >> v )
        container.push_back(v);
}

template<class DataTypes>
void create(OscillatorConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< OscillatorConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (const char* str = arg->getAttribute("indices"))
        {
            vector<unsigned> indices;
            readVector(indices,str);
            //const char* str = arg->getAttribute("indices");
            //const char* str2 = NULL;
            //for(;;)
            //{
            //	int v = (int)strtod(str,(char**)&str2);
            //	if (str2==str) break;
            //	str = str2;
            //	obj->addConstraint(v);
            //}
        }
    }
}
}

Creator< ObjectFactory, OscillatorConstraint<Vec3dTypes> > OscillatorConstraint3dClass("OscillatorConstraint",true);
Creator< ObjectFactory, OscillatorConstraint<Vec3fTypes> > OscillatorConstraint3fClass("OscillatorConstraint",true);
Creator< ObjectFactory, OscillatorConstraint<RigidTypes> > OscillatorConstraintRigidClass("OscillatorConstraint",true);

} // namespace Components

} // namespace Sofa
