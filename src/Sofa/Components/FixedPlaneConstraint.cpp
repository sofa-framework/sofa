#include "FixedPlaneConstraint.inl"
#include "Sofa/Components/Common/ObjectFactory.h"
#include "Sofa/Components/Common/Vec3Types.h"

namespace Sofa
{

namespace Components
{

using namespace Common;



SOFA_DECL_CLASS(FixedPlaneConstraint)

template class FixedPlaneConstraint<Vec3dTypes>;
template class FixedPlaneConstraint<Vec3fTypes>;

namespace Common   // \todo Why this must be inside Common namespace
{


template<class DataTypes>
void create(FixedPlaneConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    typedef typename DataTypes::Coord::value_type   Real;
    typedef typename DataTypes::Coord   Coord;

    XML::createWithParent< FixedPlaneConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("indices"))
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
        }
        if (arg->getAttribute("direction"))
        {
            const char* str = arg->getAttribute("direction");
            const char* str2 = NULL;
            Real val[3];
            unsigned int i;
            for(i=0; i<3; i++)
            {
                val[i] = (Real)strtod(str,(char**)&str2);
                if (str2==str) break;
                str = str2;
            }
            Coord dir(val);
            obj->setDirection(dir);
        }
        if (arg->getAttribute("distance"))
        {
            const char* str = arg->getAttribute("distance");
            const char* str2 = NULL;
            Real val[2];
            unsigned int i;
            for(i=0; i<2; i++)
            {
                val[i] = (Real)strtod(str,(char**)&str2);
                if (str2==str) break;
                str = str2;
            }
            obj->setDminAndDmax(val[0],val[1]);
        }
    }
}
}

Creator< ObjectFactory, FixedPlaneConstraint<Vec3dTypes> > FixedPlaneConstraint3dClass("FixedPlaneConstraint",true);
Creator< ObjectFactory, FixedPlaneConstraint<Vec3fTypes> > FixedPlaneConstraint3fClass("FixedPlaneConstraint",true);

} // namespace Components

} // namespace Sofa
