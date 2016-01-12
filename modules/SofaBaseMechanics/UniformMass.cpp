/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_MASS_UNIFORMMASS_CPP
#include <SofaBaseMechanics/UniformMass.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>
#include <sstream>

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
template<> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid3dTypes, Rigid3dMass>::reinit()
{
    if (this->totalMass.getValue()>0 && this->mstate!=NULL)
    {
        MassType* m = this->mass.beginEdit();
        *m = ((Real)this->totalMass.getValue() / mstate->getSize());
        this->mass.endEdit();
    }
    else
    {
        this->totalMass.setValue(  this->mstate->getSize()*this->mass.getValue());
    }

    this->mass.beginEdit()->recalc();
    this->mass.endEdit();
}

template<> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid3dTypes, Rigid3dMass>::loadRigidMass(std::string filename)
{
    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

//  this->totalMass.setDisplayed(false);

    if (!filename.empty())
    {
        Rigid3dMass m = this->getMass();
        if (!sofa::helper::system::DataRepository.findFile(filename))
        {
            serr << "ERROR: cannot find file '" << filename << "'." << sendl;
        }
        else
        {
            char	cmd[64];
            FILE*	file;
            if ((file = fopen(filename.c_str(), "r")) == NULL)
            {
                serr << "ERROR: cannot read file '" << filename << "'." << sendl;
            }
            else
            {
                //sout << "Loading rigid model '" << filename << "'" << sendl;
                // Check first line
                //if (fgets(cmd, 7, file) != NULL && !strcmp(cmd,"Xsp 3.0"))
                {
                    skipToEOL(file);
                    std::ostringstream cmdScanFormat;
                    cmdScanFormat << "%" << (sizeof(cmd) - 1) << "s";
                    while (fscanf(file, cmdScanFormat.str().c_str(), cmd) != EOF)
                    {
                        if (!strcmp(cmd,"inrt"))
                        {
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                    if( fscanf(file, "%lf", &(m.inertiaMatrix[i][j])) < 1 )
                                        serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"cntr") || !strcmp(cmd,"center") )
                        {
                            Vec3d center;
                            for (int i = 0; i < 3; ++i)
                            {
                                if( fscanf(file, "%lf", &(center[i])) < 1 )
                                    serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                            }
                        }
                        else if (!strcmp(cmd,"mass"))
                        {
                            double mass;
                            if( fscanf(file, "%lf", &mass) > 0 )
                            {
                                if (!this->mass.isSet())
                                    m.mass = mass;
                            }
                            else
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"volm"))
                        {
                            if( fscanf(file, "%lf", &(m.volume)) < 1 )
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"frme"))
                        {
                            Quatd orient;
                            for (int i = 0; i < 4; ++i)
                            {
                                if( fscanf(file, "%lf", &(orient[i])) < 1 )
                                    serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                            }
                            orient.normalize();
                        }
                        else if (!strcmp(cmd,"grav"))
                        {
                            Vec3d gravity;
                            if( fscanf(file, "%lf %lf %lf\n", &(gravity.x()), &(gravity.y()), &(gravity.z())) < 3 )
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"visc"))
                        {
                            double viscosity = 0;
                            if( fscanf(file, "%lf", &viscosity) < 1 )
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;

                        }
                        else if (!strcmp(cmd,"stck"))
                        {
                            double tmp;
                            if( fscanf(file, "%lf", &tmp) < 1 ) //&(MSparams.default_stick));
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"step"))
                        {
                            double tmp;
                            if( fscanf(file, "%lf", &tmp) < 1 ) //&(MSparams.default_dt));
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"prec"))
                        {
                            double tmp;
                            if( fscanf(file, "%lf", &tmp) < 1 ) //&(MSparams.default_prec));
                            {
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                            }
                        }
                        else if (cmd[0] == '#')	// it's a comment
                        {
                            skipToEOL(file);
                        }
                        else		// it's an unknown keyword
                        {
                            printf("%s: Unknown RigidMass keyword: %s\n", filename.c_str(), cmd);
                            skipToEOL(file);
                        }
                    }
                }
                fclose(file);
            }
        }
        this->setMass(m);
    }
    else if (this->totalMass.getValue()>0 && this->mstate!=NULL) this->mass.setValue((Real)this->totalMass.getValue() / mstate->getSize());

}


template <> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid3dTypes, Rigid3dMass>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
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
		if (getContext()->isSleeping())
	        vparams->drawTool()->drawFrame(x[i].getCenter(), x[i].getOrientation(), len*showAxisSize.getValue(), Vec4f(0.5,0.5,0.5,1) );
		else
			vparams->drawTool()->drawFrame(x[i].getCenter(), x[i].getOrientation(), len*showAxisSize.getValue() );
        gravityCenter += (x[i].getCenter());
    }

    if (showInitialCenterOfGravity.getValue())
    {
        const VecCoord& x0 = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        for (unsigned int i=0; i<x0.size(); i++)
        {
            vparams->drawTool()->drawFrame(x0[i].getCenter(), x0[i].getOrientation(), len*showAxisSize.getValue());
        }
    }

    if(showCenterOfGravity.getValue())
    {
        gravityCenter /= x.size();
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

        vparams->drawTool()->drawCross(gravityCenter, showAxisSize.getValue(), color);
    }
}



template <> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid2dTypes, Rigid2dMass>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    defaulttype::Vec3d len;

    len[0] = len[1] = sqrt(mass.getValue().inertiaMatrix);
    len[2] = 0;

    for (unsigned int i=0; i<x.size(); i++)
    {
        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();

        vparams->drawTool()->drawFrame(center, orient, len*showAxisSize.getValue() );
    }
}

template <> SOFA_BASE_MECHANICS_API
SReal UniformMass<Rigid3dTypes,Rigid3dMass>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx ) const
{
    SReal e = 0;
    helper::ReadAccessor< DataVecCoord > x = vx;
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}


template <> SOFA_BASE_MECHANICS_API
SReal UniformMass<Rigid2dTypes,Rigid2dMass>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx ) const
{
    SReal e = 0;
    helper::ReadAccessor< DataVecCoord > x = vx;
    // gravity
    Vec2d g; g = this->getContext()->getGravity();
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}



template <> SOFA_BASE_MECHANICS_API
void UniformMass<Vec6dTypes, double>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x0 = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    Mat3x3d R; R.identity();

    std::vector<Vector3> vertices;
    std::vector<sofa::defaulttype::Vec4f> colors;

    const sofa::defaulttype::Vec4f red(1.0,0.0,0.0,1.0);
    const sofa::defaulttype::Vec4f green(0.0,1.0,0.0,1.0);
    const sofa::defaulttype::Vec4f blue(0.0,0.0,1.0,1.0);

    sofa::defaulttype::Vec4f colorSet[3];
    colorSet[0] = red;
    colorSet[1] = green;
    colorSet[2] = blue;

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

        for(unsigned int j=0 ; j<3 ; j++)
        {
            vertices.push_back(p);
            vertices.push_back(p + R.col(j)*len[j]);
            colors.push_back(colorSet[j]);
            colors.push_back(colorSet[j]);;
        }
    }
}

template <> SOFA_BASE_MECHANICS_API
void UniformMass<Vec3dTypes, double>::addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv* dx, SReal mFact, unsigned int& offset)
{
    unsigned int derivDim = (unsigned)Deriv::size();
    double m = mass.getValue();

    unsigned int vecDim = (unsigned)mstate->getSize();

    const double* g = this->getContext()->getGravity().ptr();

    for (unsigned int i=0; i<vecDim; i++)
        for (unsigned int j=0; j<derivDim; j++)
        {
            if (dx != NULL)
                resVect->add(offset + i * derivDim + j, mFact * m * g[j] * (*dx)[i][0]);
            else
                resVect->add(offset + i * derivDim + j, mFact * m * g[j]);
        }
}

template <> SOFA_BASE_MECHANICS_API
Vector6 UniformMass<Vec3dTypes, double>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = (unsigned)v.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    const MassType& m = mass.getValue();

    defaulttype::Vec6d momentum;

    for ( unsigned int i=ibegin ; i<iend ; i++ )
    {
        Deriv linearMomentum = m*v[i];
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

template <> SOFA_BASE_MECHANICS_API
Vector6 UniformMass<Rigid3dTypes,Rigid3dMass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = (unsigned)v.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    Real m = mass.getValue().mass;
    const Rigid3dMass::Mat3x3& I = mass.getValue().inertiaMassMatrix;

    defaulttype::Vec6d momentum;

    for ( unsigned int i=ibegin ; i<iend ; i++ )
    {
        Rigid3dTypes::Vec3 linearMomentum = m*v[i].getLinear();
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Rigid3dTypes::Vec3 angularMomentum = cross( x[i].getCenter(), linearMomentum ) + ( I * v[i].getAngular() );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

#endif

#ifndef SOFA_DOUBLE
template<> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid3fTypes, Rigid3fMass>::reinit()
{
    if (this->totalMass.getValue()>0 && this->mstate!=NULL)
    {
        MassType* m = this->mass.beginEdit();
        *m = ((Real)this->totalMass.getValue() / mstate->getSize());
        this->mass.endEdit();
    }
    else
    {
        this->totalMass.setValue(  this->mstate->getSize()*this->mass.getValue());
    }

    this->mass.beginEdit()->recalc();
    this->mass.endEdit();
}

template<> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid3fTypes, Rigid3fMass>::loadRigidMass(std::string filename)
{
    this->totalMass.setDisplayed(false);
    if (!filename.empty())
    {
        Rigid3fMass m = this->getMass();
        if (!sofa::helper::system::DataRepository.findFile(filename))
        {
            serr << "ERROR: cannot find file '" << filename << "'." << sendl;
        }
        else
        {
            char	cmd[64];
            FILE*	file;
            if ((file = fopen(filename.c_str(), "r")) == NULL)
            {
                serr << "ERROR: cannot read file '" << filename << "'." << sendl;
            }
            else
            {
                //sout << "Loading rigid model '" << filename << "'" << sendl;
                // Check first line
                //if (fgets(cmd, 7, file) != NULL && !strcmp(cmd,"Xsp 3.0"))
                {
                    skipToEOL(file);
                    std::ostringstream cmdScanFormat;
                    cmdScanFormat << "%" << (sizeof(cmd) - 1) << "s";
                    while (fscanf(file, cmdScanFormat.str().c_str(), cmd) != EOF)
                    {
                        if (!strcmp(cmd,"inrt"))
                        {
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                    if( fscanf(file, "%f", &(m.inertiaMatrix[i][j])) < 1 )
                                        serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;

                        }
                        else if (!strcmp(cmd,"cntr"))
                        {
                            Vec3d center;
                            for (int i = 0; i < 3; ++i)
                            {
                                if( fscanf(file, "%lf", &(center[i])) < 1 )
                                    serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                            }
                        }
                        else if (!strcmp(cmd,"mass"))
                        {
                            float mass;
                            if( fscanf(file, "%f", &mass) > 0 )
                            {
                                if (!this->mass.isSet())
                                    m.mass = mass;
                            }
                            else
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"volm"))
                        {
                            if( fscanf(file, "%f", &(m.volume)) < 1 )
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"frme"))
                        {
                            Quatd orient;
                            for (int i = 0; i < 4; ++i)
                            {
                                if( fscanf(file, "%lf", &(orient[i])) < 1 )
                                    serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                            }
                            orient.normalize();
                        }
                        else if (!strcmp(cmd,"grav"))
                        {
                            Vec3d gravity;
                            if( fscanf(file, "%lf %lf %lf\n", &(gravity.x()), &(gravity.y()), &(gravity.z())) < 3 )
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"visc"))
                        {
                            double viscosity = 0;
                            if( fscanf(file, "%lf", &viscosity) < 1 )
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"stck"))
                        {
                            double tmp;
                            if( fscanf(file, "%lf", &tmp) < 1 ) //&(MSparams.default_stick));
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"step"))
                        {
                            double tmp;
                            if( fscanf(file, "%lf", &tmp) < 1 ) //&(MSparams.default_dt));
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (!strcmp(cmd,"prec"))
                        {
                            double tmp;
                            if( fscanf(file, "%lf", &tmp) < 1 ) //&(MSparams.default_prec));
                                serr << SOFA_CLASS_METHOD << "error reading file '" << filename << "'." << sendl;
                        }
                        else if (cmd[0] == '#')	// it's a comment
                        {
                            skipToEOL(file);
                        }
                        else		// it's an unknown keyword
                        {
                            printf("%s: Unknown RigidMass keyword: %s\n", filename.c_str(), cmd);
                            skipToEOL(file);
                        }
                    }
                }
                fclose(file);
            }
        }

        this->setMass(m);
    }
    else if (this->totalMass.getValue()>0 ) this->mass.setValue((Real)this->totalMass.getValue());
    this->totalMass.setValue(0.0f);

}


template <> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid3fTypes, Rigid3fMass>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
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
        vparams->drawTool()->drawFrame(x[i].getCenter(), x[i].getOrientation(), len*showAxisSize.getValue() );
        gravityCenter += (x[i].getCenter());
    }

    if(showCenterOfGravity.getValue())
    {
        gravityCenter /= x.size();
        const sofa::defaulttype::Vec4f color(1.0,1.0,0.0,1.0);

//        sofa::defaulttype::Vec3f temp = gravityCenter;
//        for(unsigned int i=0 ; i<3 ; i++)
//            temp[i] = gravityCenter[i];

        vparams->drawTool()->drawCross(gravityCenter, showAxisSize.getValue(), color);
    }
}

template <> SOFA_BASE_MECHANICS_API
void UniformMass<Rigid2fTypes, Rigid2fMass>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    defaulttype::Vec3d len;

    len[0] = len[1] = sqrt(mass.getValue().inertiaMatrix);
    len[2] = 0;

    for (unsigned int i=0; i<x.size(); i++)
    {
        Quat orient(Vec3d(0,0,1), x[i].getOrientation());
        Vec3d center; center = x[i].getCenter();

        vparams->drawTool()->drawFrame(center, orient, len*showAxisSize.getValue() );
    }
}

template <> SOFA_BASE_MECHANICS_API
SReal UniformMass<Rigid3fTypes,Rigid3fMass>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx ) const
{
    SReal e = 0;
    helper::ReadAccessor< DataVecCoord > x = vx;
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <> SOFA_BASE_MECHANICS_API
SReal UniformMass<Rigid2fTypes,Rigid2fMass>::getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& vx) const
{
    SReal e = 0;
    helper::ReadAccessor< DataVecCoord > x = vx;
    // gravity
    Vec2d g; g = this->getContext()->getGravity();
    for (unsigned int i=0; i<x.size(); i++)
    {
        e -= g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}



template <> SOFA_BASE_MECHANICS_API
void UniformMass<Vec6fTypes, float>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x =mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x0 = mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    Mat3x3d R;

    std::vector<Vector3> vertices;
    std::vector<sofa::defaulttype::Vec4f> colors;

    const sofa::defaulttype::Vec4f red(1.0,0.0,0.0,1.0);
    const sofa::defaulttype::Vec4f green(0.0,1.0,0.0,1.0);
    const sofa::defaulttype::Vec4f blue(0.0,0.0,1.0,1.0);

    sofa::defaulttype::Vec4f colorSet[3];
    colorSet[0] = red;
    colorSet[1] = green;
    colorSet[2] = blue;

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

        for(unsigned int j=0 ; j<3 ; j++)
        {
            vertices.push_back(p);
            vertices.push_back(p + R.col(j)*len[j]);
            colors.push_back(colorSet[j]);
            colors.push_back(colorSet[j]);;
        }
    }
}


template <> SOFA_BASE_MECHANICS_API
void UniformMass<Vec3fTypes, float>::addMDxToVector(defaulttype::BaseVector *resVect, const VecDeriv* dx, SReal mFact, unsigned int& offset)
{
    unsigned int derivDim = (unsigned)Deriv::size();
    float m = mass.getValue();

    unsigned int vecDim = (unsigned)mstate->getSize();

    const SReal* g = this->getContext()->getGravity().ptr();

    for (unsigned int i=0; i<vecDim; i++)
        for (unsigned int j=0; j<derivDim; j++)
        {
            if (dx != NULL)
                resVect->add(offset + i * derivDim + j, mFact * m * g[j] * (*dx)[i][0]);
            else
                resVect->add(offset + i * derivDim + j, mFact * m * g[j]);
        }
}


template <> SOFA_BASE_MECHANICS_API
Vector6 UniformMass<Vec3fTypes, float>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = (unsigned)v.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    const MassType& m = mass.getValue();

    defaulttype::Vec6d momentum;

    for ( unsigned int i=ibegin ; i<iend ; i++ )
    {
        Deriv linearMomentum = m*v[i];
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Deriv angularMomentum = cross( x[i], linearMomentum );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}

template <> SOFA_BASE_MECHANICS_API
Vector6 UniformMass<Rigid3fTypes,Rigid3fMass>::getMomentum ( const core::MechanicalParams*, const DataVecCoord& vx, const DataVecDeriv& vv ) const
{
    helper::ReadAccessor<DataVecDeriv> v = vv;
    helper::ReadAccessor<DataVecCoord> x = vx;

    unsigned int ibegin = 0;
    unsigned int iend = (unsigned)v.size();

    if ( localRange.getValue() [0] >= 0 )
        ibegin = localRange.getValue() [0];

    if ( localRange.getValue() [1] >= 0 && ( unsigned int ) localRange.getValue() [1]+1 < iend )
        iend = localRange.getValue() [1]+1;

    Real m = mass.getValue().mass;
    const Rigid3fMass::Mat3x3& I = mass.getValue().inertiaMassMatrix;

    defaulttype::Vec6d momentum;

    for ( unsigned int i=ibegin ; i<iend ; i++ )
    {
        Rigid3fTypes::Vec3 linearMomentum = m*v[i].getLinear();
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[j] += linearMomentum[j];

        Rigid3fTypes::Vec3 angularMomentum = cross( x[i].getCenter(), linearMomentum ) + ( I * v[i].getAngular() );
        for( int j=0 ; j<DataTypes::spatial_dimensions ; ++j ) momentum[3+j] += angularMomentum[j];
    }

    return momentum;
}


#endif


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
template class SOFA_BASE_MECHANICS_API UniformMass<Vec3dTypes,double>;
template class SOFA_BASE_MECHANICS_API UniformMass<Vec2dTypes,double>;
template class SOFA_BASE_MECHANICS_API UniformMass<Vec1dTypes,double>;
template class SOFA_BASE_MECHANICS_API UniformMass<Vec6dTypes,double>;
template class SOFA_BASE_MECHANICS_API UniformMass<Rigid3dTypes,Rigid3dMass>;
template class SOFA_BASE_MECHANICS_API UniformMass<Rigid2dTypes,Rigid2dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API UniformMass<Vec3fTypes,float>;
template class SOFA_BASE_MECHANICS_API UniformMass<Vec2fTypes,float>;
template class SOFA_BASE_MECHANICS_API UniformMass<Vec1fTypes,float>;
template class SOFA_BASE_MECHANICS_API UniformMass<Vec6fTypes,float>;
template class SOFA_BASE_MECHANICS_API UniformMass<Rigid3fTypes,Rigid3fMass>;
template class SOFA_BASE_MECHANICS_API UniformMass<Rigid2fTypes,Rigid2fMass>;
#endif

} // namespace mass

} // namespace component

} // namespace sofa
