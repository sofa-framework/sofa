#include "FixedConstraint.inl"
#include "Sofa/Components/Common/ObjectFactory.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <>
void FixedConstraint<RigidTypes>::draw()
{
    /*	if (!getContext()->getShowBehaviorModels()) return;
    	VecCoord& x = *mmodel->getX();
    	for (std::set<int>::const_iterator it = this->indices.begin(); it != this->indices.end(); ++it)
    	{
    		int i = *it;
    		Quat orient = x[i].getOrientation();
    		RigidTypes::Vec3& center = x[i].getCenter();
    		orient[3] = -orient[3];

    		static GL::Axis *axis = new GL::Axis(center, orient, 5);

    		axis->update(center, orient);
    		axis->draw();
    	}*/
    const SetIndex& indices = f_indices.getValue();
    if (!getContext()->getShowBehaviorModels()) return;
    VecCoord& x = *mmodel->getX();
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_POINTS);
    for (vector<int>::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        GL::glVertexT(x[0].getCenter());
    }
    glEnd();
}

template <>
void FixedConstraint<RigidTypes>::projectResponse(VecDeriv& res)
{
    res[0] = Deriv();
}


SOFA_DECL_CLASS(FixedConstraint)

template class FixedConstraint<Vec3dTypes>;
template class FixedConstraint<Vec3fTypes>;

namespace Common   // \todo Why this must be inside Common namespace
{

template<class DataTypes>
void create(FixedConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< FixedConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
        /*		if (arg->getAttribute("indices"))
        		{
        			const char* str = arg->getAttribute("indices");
        			const char* str2 = NULL;
        			for(;;)
        			{
        				int v = (int)strtod(str,(char**)&str2);
        				if (str2==str) break;
        				str = str2;
        				obj->addConstraint(v);
        			}
        		}*/
    }
}
}

Creator< ObjectFactory, FixedConstraint<Vec3dTypes> > FixedConstraint3dClass("FixedConstraint",true);
Creator< ObjectFactory, FixedConstraint<Vec3fTypes> > FixedConstraint3fClass("FixedConstraint",true);
Creator< ObjectFactory, FixedConstraint<RigidTypes> > FixedConstraintRigidClass("FixedConstraint",true);

} // namespace Components

} // namespace Sofa
