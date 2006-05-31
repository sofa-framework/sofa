#include "UniformMass.inl"
#include "Common/ObjectFactory.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/RigidTypes.h"
#include "GL/Repere.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template <>
void UniformMass<RigidTypes, RigidMass>::draw()
{
    if (!getContext()->getShowBehaviorModels()) return;
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

SOFA_DECL_CLASS(UniformMass)

template class UniformMass<Vec3dTypes,double>;
template class UniformMass<Vec3fTypes,float>;
template class UniformMass<RigidTypes,RigidMass>;

namespace Common   // \todo Why this must be inside Common namespace
{

template<class DataTypes, class MassType>
void create(UniformMass<DataTypes, MassType>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< UniformMass<DataTypes, MassType>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("mass"))
        {
            obj->setMass((MassType)atof(arg->getAttribute("mass")));
        }
    }
}

static void skipToEOL(FILE* f)
{
    int	ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n');
}

template<>
void create(UniformMass<RigidTypes, RigidMass>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< UniformMass<RigidTypes, RigidMass>, Core::MechanicalModel<RigidTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("mass"))
        {
            obj->setMass(RigidMass(atof(arg->getAttribute("mass"))));
        }
        if (arg->getAttribute("filename"))
        {
            RigidMass m;
            const char* filename = arg->getAttribute("filename");
            char	cmd[64];
            FILE*	file;
            if ((file = fopen(filename, "r")) == NULL)
            {
                std::cerr << "ERROR: cannot read file '" << filename << "'." << std::endl;
            }
            else
            {
                std::cout << "Loading rigid model '" << filename << "'" << std::endl;
                // Check first line
                //if (fgets(cmd, 7, file) != NULL && !strcmp(cmd,"Xsp 3.0"))
                {
                    skipToEOL(file);

                    while (fscanf(file, "%s", cmd) != EOF)
                    {
                        if (!strcmp(cmd,"inrt"))
                        {
                            for (int i = 0; i < 9; i++)
                            {
                                fscanf(file, "%lf", &(m.inertiaMatrix.ptr()[i]));
                            }
                        }
                        else if (!strcmp(cmd,"cntr"))
                        {
                            Vec3d center;
                            for (int i = 0; i < 3; ++i)
                            {
                                fscanf(file, "%lf", &(center[i]));
                            }
                        }
                        else if (!strcmp(cmd,"mass"))
                        {
                            fscanf(file, "%lf", &(m.mass));
                        }
                        else if (!strcmp(cmd,"frme"))
                        {
                            Quat orient;
                            for (int i = 0; i < 4; ++i)
                            {
                                fscanf(file, "%lf", &(orient[i]));
                            }
                            orient.normalize();
                        }
                        else if (!strcmp(cmd,"grav"))
                        {
                            Vec3d gravity;
                            fscanf(file, "%lf %lf %lf\n", &(gravity.x()),
                                    &(gravity.y()), &(gravity.z()));
                        }
                        else if (!strcmp(cmd,"visc"))
                        {
                            double viscosity = 0;
                            fscanf(file, "%lf", &viscosity);
                        }
                        else if (!strcmp(cmd,"stck"))
                        {
                            double tmp;
                            fscanf(file, "%lf", &tmp); //&(MSparams.default_stick));
                        }
                        else if (!strcmp(cmd,"step"))
                        {
                            double tmp;
                            fscanf(file, "%lf", &tmp); //&(MSparams.default_dt));
                        }
                        else if (!strcmp(cmd,"prec"))
                        {
                            double tmp;
                            fscanf(file, "%lf", &tmp); //&(MSparams.default_prec));
                        }
                        else if (cmd[0] == '#')	// it's a comment
                        {
                            skipToEOL(file);
                        }
                        else		// it's an unknown keyword
                        {
                            printf("Unknown keyword: %s\n", cmd);
                            skipToEOL(file);
                        }
                    }
                }
                fclose(file);
            }
            m.recalc();
            obj->setMass(m);
        }
    }
}

}

Creator< ObjectFactory, UniformMass<Vec3dTypes,double> > UniformMass3dClass("UniformMass",true);
Creator< ObjectFactory, UniformMass<Vec3fTypes,float > > UniformMass3fClass("UniformMass",true);
Creator< ObjectFactory, UniformMass<RigidTypes,RigidMass> > UniformMassRigidClass("UniformMass",true);

} // namespace Components

} // namespace Sofa
