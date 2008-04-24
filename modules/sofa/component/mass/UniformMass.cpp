/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/mass/UniformMass.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mass
{



using namespace sofa::defaulttype;


static void skipToEOL(FILE* f)
{
    int	ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n')
        ;
}


Mat3x3d MatrixFromEulerXYZ(double thetaX, double thetaY, double thetaZ)
{
    double cosX = cos(thetaX);
    double sinX = sin(thetaX);
    double cosY = cos(thetaY);
    double sinY = sin(thetaY);
    double cosZ = cos(thetaZ);
    double sinZ = sin(thetaZ);
    return
        Mat3x3d(Vec3d( cosZ, -sinZ,     0),
                Vec3d( sinZ,  cosZ,     0),
                Vec3d(    0,     0,     1)) *
        Mat3x3d(Vec3d( cosY,     0,  sinY),
                Vec3d(    0,     1,     0),
                Vec3d(-sinY,     0,  cosY)) *
        Mat3x3d(Vec3d(    1,     0,     0),
                Vec3d(    0,  cosX, -sinX),
                Vec3d(    0,  sinX,  cosX)) ;
}




#ifndef SOFA_FLOAT
template<>
void UniformMass<Rigid3dTypes, Rigid3dMass>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherited::parse(arg);
    Rigid3dMass m = this->getMass();
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
            //std::cout << "Loading rigid model '" << filename << "'" << std::endl;
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
                        double mass;
                        fscanf(file, "%lf", &mass);
                        if (!arg->getAttribute("mass"))
                            m.mass = mass;
                    }
                    else if (!strcmp(cmd,"volm"))
                    {
                        fscanf(file, "%lf", &(m.volume));
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
    m.recalc();
    this->setMass(m);
}


template <>
void UniformMass<Rigid3dTypes, Rigid3dMass>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    RigidTypes::Vec3 gravityCenter;
    defaulttype::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = mass.getValue().inertiaMatrix[0][0];
    double m11 = mass.getValue().inertiaMatrix[1][1];
    double m22 = mass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len*showAxisSize.getValue());
        gravityCenter += (x[i].getCenter());
    }

    if(showCenterOfGravity.getValue())
    {
        gravityCenter /= x.size();
        glColor3f (1,1,0);
        glBegin (GL_LINES);
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        glEnd();
    }
}



template <>
void UniformMass<Rigid2dTypes, Rigid2dMass>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    defaulttype::Vec3d len;

    len[0] = len[1] = sqrt(mass.getValue().inertiaMatrix);
    len[2] = 0;

    for (unsigned int i=0; i<x.size(); i++)
    {
        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();
        helper::gl::Axis::draw(center, orient, len);
    }
}

template <>
double UniformMass<Rigid3dTypes,Rigid3dMass>::getPotentialEnergy( const Rigid3dTypes::VecCoord& x )
{
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}


template <>
double UniformMass<Rigid2dTypes,Rigid2dMass>::getPotentialEnergy( const Rigid2dTypes::VecCoord& x )
{
    double e = 0;
    // gravity
    Vec2d g; g = this->getContext()->getLocalGravity();
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid3dTypes,Rigid3dMass>::getElementMass(unsigned int )
{
    return (mass.getValue().mass);
}

template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid2dTypes,Rigid2dMass>::getElementMass(unsigned int )
{
    return (mass.getValue().mass);
}


template <>
void UniformMass<Vec6dTypes, double>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    const VecCoord& x0 = *mstate->getX0();

    Mat3x3d R; R.identity();
    glBegin(GL_LINES);
    for (unsigned int i=0; i<x.size(); i++)
    {
        defaulttype::Vec3d len(1,1,1);
        int a = (i<x.size()-1)?i : i-1;
        int b = a+1;
        defaulttype::Vec3d dp; dp = x0[b]-x0[a];
        defaulttype::Vec3d p; p = x[i];
        len[0] = dp.norm();
        len[1] = len[0];
        len[2] = len[0];
        R = R * MatrixFromEulerXYZ(x[i][3], x[i][4], x[i][5]);
        glColor3f(1,0,0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p + R.col(0)*len[0]);
        glColor3f(0,1,0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p + R.col(1)*len[1]);
        glColor3f(0,0,1);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p + R.col(2)*len[2]);
    }
    glEnd();
}

template <>
void UniformMass<Vec3dTypes, double>::addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv* dx, double mFact, unsigned int& offset)
{
    unsigned int derivDim = Deriv::size();
    double m = mass.getValue();

    unsigned int vecDim = mstate->getX()->size();

    const double* g = this->getContext()->getLocalGravity().ptr();

    for (unsigned int i=0; i<vecDim; i++)
        for (unsigned int j=0; j<derivDim; j++)
        {
            if (dx != NULL)
                resVect->add(offset + i * derivDim + j, mFact * m * g[j] * (*dx)[i][0]);
            else
                resVect->add(offset + i * derivDim + j, mFact * m * g[j]);
        }
}

#endif

#ifndef SOFA_DOUBLE
template<>
void UniformMass<Rigid3fTypes, Rigid3fMass>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherited::parse(arg);
    Rigid3fMass m = this->getMass();
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
            //std::cout << "Loading rigid model '" << filename << "'" << std::endl;
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
                            fscanf(file, "%f", &(m.inertiaMatrix.ptr()[i]));
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
                        float mass;
                        fscanf(file, "%f", &mass);
                        if (!arg->getAttribute("mass"))
                            m.mass = mass;
                    }
                    else if (!strcmp(cmd,"volm"))
                    {
                        fscanf(file, "%f", &(m.volume));
                    }
                    else if (!strcmp(cmd,"frme"))
                    {
                        Quatd orient;
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
    m.recalc();
    this->setMass(m);
}


template <>
void UniformMass<Rigid3fTypes, Rigid3fMass>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    RigidTypes::Vec3 gravityCenter;
    defaulttype::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = mass.getValue().inertiaMatrix[0][0];
    double m11 = mass.getValue().inertiaMatrix[1][1];
    double m22 = mass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len*showAxisSize.getValue());
        gravityCenter += (x[i].getCenter());
    }

    if(showCenterOfGravity.getValue())
    {
        gravityCenter /= x.size();
        glColor3f (1,1,0);
        glBegin (GL_LINES);
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(showAxisSize.getValue(),0,0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,showAxisSize.getValue(),0) );
        helper::gl::glVertexT(gravityCenter - RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        helper::gl::glVertexT(gravityCenter + RigidTypes::Vec3(0,0,showAxisSize.getValue()) );
        glEnd();
    }
}

template <>
void UniformMass<Rigid2fTypes, Rigid2fMass>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    defaulttype::Vec3d len;

    len[0] = len[1] = sqrt(mass.getValue().inertiaMatrix);
    len[2] = 0;

    for (unsigned int i=0; i<x.size(); i++)
    {
        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();
        helper::gl::Axis::draw(center, orient, len);
    }
}

template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid3fTypes,Rigid3fMass>::getPotentialEnergy( const Rigid3fTypes::VecCoord& x )
{
    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getLocalGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid2fTypes,Rigid2fMass>::getPotentialEnergy( const Rigid2fTypes::VecCoord& x )
{
    double e = 0;
    // gravity
    Vec2d g; g = this->getContext()->getLocalGravity();
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid3fTypes,Rigid3fMass>::getElementMass(unsigned int )
{
    return (sofa::defaulttype::Vector3::value_type)(mass.getValue().mass);
}


template <>
sofa::defaulttype::Vector3::value_type UniformMass<Rigid2fTypes,Rigid2fMass>::getElementMass(unsigned int )
{
    return (sofa::defaulttype::Vector3::value_type)(mass.getValue().mass);
}

template <>
void UniformMass<Vec6fTypes, float>::draw()
{
    if (!getContext()->getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    const VecCoord& x0 = *mstate->getX0();

    Mat3x3d R;
    glBegin(GL_LINES);
    for (unsigned int i=0; i<x.size(); i++)
    {
        defaulttype::Vec3d len(1,1,1);
        int a = (i<x.size()-1)?i : i-1;
        int b = a+1;
        defaulttype::Vec3d dp; dp = x0[b]-x0[a];
        defaulttype::Vec3d p; p = x[i];
        len[0] = dp.norm();
        len[1] = len[0];
        len[2] = len[0];
        R = R * MatrixFromEulerXYZ(x[i][3], x[i][4], x[i][5]);
        glColor3f(1,0,0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p + R.col(0)*len[0]);
        glColor3f(0,1,0);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p + R.col(1)*len[1]);
        glColor3f(0,0,1);
        helper::gl::glVertexT(p);
        helper::gl::glVertexT(p + R.col(2)*len[2]);
    }
    glEnd();
}


template <>
void UniformMass<Vec3fTypes, float>::addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv* dx, Real_Sofa mFact, unsigned int& offset)
{
    unsigned int derivDim = Deriv::size();
    float m = mass.getValue();

    unsigned int vecDim = mstate->getX()->size();

    const Real_Sofa* g = this->getContext()->getLocalGravity().ptr();

    for (unsigned int i=0; i<vecDim; i++)
        for (unsigned int j=0; j<derivDim; j++)
        {
            if (dx != NULL)
                resVect->add(offset + i * derivDim + j, mFact * m * g[j] * (*dx)[i][0]);
            else
                resVect->add(offset + i * derivDim + j, mFact * m * g[j]);
        }
}

#endif

// specialization for rigid bodies


// specialization for rigid bodies

// specialization for rigid bodies
// specialization for rigid bodies






SOFA_DECL_CLASS(UniformMass)

// Register in the Factory
int UniformMassClass = core::RegisterObject("Define the same mass for all the particles")

#ifndef SOFA_FLOAT
        .add< UniformMass<Vec3dTypes,double> >()
        .add< UniformMass<Vec2dTypes,double> >()
        .add< UniformMass<Vec1dTypes,double> >()
        .add< UniformMass<Vec6dTypes,double> >()
        .add< UniformMass<Rigid3dTypes,Rigid3dMass> >()
        .add< UniformMass<Rigid2dTypes,Rigid2dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< UniformMass<Vec3fTypes,float> >()
        .add< UniformMass<Vec2fTypes,float> >()
        .add< UniformMass<Vec1fTypes,float> >()
        .add< UniformMass<Vec6fTypes,float> >()
        .add< UniformMass<Rigid3fTypes,Rigid3fMass> >()
        .add< UniformMass<Rigid2fTypes,Rigid2fMass> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class UniformMass<Vec3dTypes,double>;
template class UniformMass<Vec2dTypes,double>;
template class UniformMass<Vec1dTypes,double>;
template class UniformMass<Vec6dTypes,double>;
template class UniformMass<Rigid3dTypes,Rigid3dMass>;
template class UniformMass<Rigid2dTypes,Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
template class UniformMass<Vec3fTypes,float>;
template class UniformMass<Vec2fTypes,float>;
template class UniformMass<Vec1fTypes,float>;
template class UniformMass<Vec6fTypes,float>;
template class UniformMass<Rigid3fTypes,Rigid3fMass>;
template class UniformMass<Rigid2fTypes,Rigid2fMass>;
#endif

} // namespace mass

} // namespace component

} // namespace sofa
