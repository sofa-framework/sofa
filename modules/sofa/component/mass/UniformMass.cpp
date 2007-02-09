#include <sofa/component/mass/UniformMass.inl>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{

namespace mass
{




static void skipToEOL(FILE* f)
{
    int	ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n')
        ;
}

template<>
void UniformMass<RigidTypes, RigidMass>::parse (core::objectmodel::BaseObjectDescription* arg)
{
    RigidMass m(1.0f);
    if (arg->getAttribute("filename"))
    {
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
                        std::cout << "mass="<<m.mass<<"\n";
                    }
                    else if (!strcmp(cmd,"volm"))
                    {
                        fscanf(file, "%lf", &(m.volume));
                        std::cout << "volm="<<m.volume<<"\n";
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
                        printf("%s: Unknown RigidMass keyword: %s\n", filename, cmd);
                        skipToEOL(file);
                    }
                }
            }
            fclose(file);
        }
    }
    if (arg->getAttribute("mass"))
    {
        m.mass = atof(arg->getAttribute("mass"));
    }
    if (arg->getAttribute("totalmass"))
    {
        m.mass = atof(arg->getAttribute("totalmass"));
    }
    m.recalc();
    this->setMass(m);
}



using namespace sofa::defaulttype;

template <>
void UniformMass<RigidTypes, RigidMass>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    VecCoord& x = *mstate->getX();
    RigidTypes::Vec3 len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = mass.inertiaMatrix[0][0];
    double m11 = mass.inertiaMatrix[1][1];
    double m22 = mass.inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        const Quat& orient = x[i].getOrientation();
        const RigidTypes::Vec3& center = x[i].getCenter();
        //orient[3] = -orient[3];

        helper::gl::Axis::draw(center, orient, len);
    }
}

// specialization for rigid bodies
template <>
double UniformMass<RigidTypes,RigidMass>::getPotentialEnergy( const RigidTypes::VecCoord& x )
{
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    Deriv theGravity;
    DataTypes::set
    ( theGravity, g[0], g[1], g[2]);
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += g*mass.mass*x[i].getCenter();
    }
    return e;
}

SOFA_DECL_CLASS(UniformMass)

template class UniformMass<Vec3dTypes,double>
;
template class UniformMass<Vec3fTypes,float>
;
template class UniformMass<RigidTypes,RigidMass>
;

// Register in the Factory
int UniformMassClass = core::RegisterObject("TODO")
        .add< UniformMass<Vec3dTypes,double> >()
        .add< UniformMass<Vec3fTypes,float> >()
        .add< UniformMass<RigidTypes,RigidMass> >()
        ;


// using helper::Creator;
//
// Creator<simulation::tree::xml::ObjectFactory, UniformMass<Vec3dTypes,double> > UniformMass3dClass("UniformMass",true);
// Creator<simulation::tree::xml::ObjectFactory, UniformMass<Vec3fTypes,float > > UniformMass3fClass("UniformMass",true);
// Creator<simulation::tree::xml::ObjectFactory, UniformMass<RigidTypes,RigidMass> > UniformMassRigidClass("UniformMass",true);

} // namespace mass

} // namespace component

} // namespace sofa
