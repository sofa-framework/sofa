/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_INL
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_INL

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/Simulation.h>
#include <SofaBoundaryCondition/PlaneForceField.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>


namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
void PlaneForceField<DataTypes>::setPlane(const Deriv& normal, Real d)
{
	DPos tmpN = DataTypes::getDPos(normal);
	Real n = tmpN.norm();
	planeNormal.setValue( tmpN / n);
	planeD.setValue( d / n );
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v1 = v;

    //this->dfdd.resize(p1.size());
    this->contacts.clear();
    f1.resize(p1.size());

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

	Real limit = this->maxForce.getValue();
	limit *= limit; // squared

	Real stiff = this->stiffness.getValue();
	Real damp = this->damping.getValue();
	DPos planeN = planeNormal.getValue();

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = DataTypes::getCPos(p1[i])*planeN-planeD.getValue();
        if (bilateral.getValue() || d<0 )
        {
            //serr<<"PlaneForceField<DataTypes>::addForce, d = "<<d<<sendl;
            Real forceIntensity = -stiff*d;
            //serr<<"PlaneForceField<DataTypes>::addForce, stiffness = "<<stiffness.getValue()<<sendl;
            Real dampingIntensity = -damp*d;
            //serr<<"PlaneForceField<DataTypes>::addForce, dampingIntensity = "<<dampingIntensity<<sendl;
            DPos force = planeN*forceIntensity - DataTypes::getDPos(v1[i])*dampingIntensity; 

			Real amplitude = force.norm2();
			if(limit && amplitude > limit)
				force *= sqrt(limit / amplitude);
            //serr<<"PlaneForceField<DataTypes>::addForce, force = "<<force<<sendl;
			Deriv tmpF;
			DataTypes::setDPos(tmpF, force);
            f1[i] += tmpF;
            //this->dfdd[i] = -this->stiffness;
            this->contacts.push_back(i);
        }
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;

    df1.resize(dx1.size());
    const Real fact = (Real)(-this->stiffness.getValue() * mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()));
	DPos planeN = planeNormal.getValue();

    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        assert(p<dx1.size());
        DataTypes::setDPos(df1[p], DataTypes::getDPos(df1[p]) + planeN * (fact * (DataTypes::getDPos(dx1[p]) * planeN)));
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    const Real fact = (Real)(-this->stiffness.getValue()*mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()));
    Deriv normal;
	DataTypes::setDPos(normal, planeNormal.getValue());
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::defaulttype::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;

    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        for (int l=0; l<Deriv::total_size; ++l)
            for (int c=0; c<Deriv::total_size; ++c)
            {
                SReal coef = normal[l] * fact * normal[c];
                mat->add(offset + p*Deriv::total_size + l, offset + p*Deriv::total_size + c, coef);
            }
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::updateStiffness( const VecCoord& vx )
{
    helper::ReadAccessor<VecCoord> x = vx;

    this->contacts.clear();

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = DataTypes::getCPos(x[i])*planeNormal.getValue()-planeD.getValue();
        if (d<0)
            this->contacts.push_back(i);
    }
}


// Rotate the plane. Note that the rotation is only applied on the 3 first coordinates
template<class DataTypes>
void PlaneForceField<DataTypes>::rotate( Deriv axe, Real angle )
{
    defaulttype::Vec3d axe3d(1,1,1); axe3d = DataTypes::getDPos(axe);
    defaulttype::Vec3d normal3d; normal3d = planeNormal.getValue();
    defaulttype::Vec3d v = normal3d.cross(axe3d);
    if (v.norm2() < 1.0e-10) return;
    v.normalize();
    v = normal3d * cos ( angle ) + v * sin ( angle );
    *planeNormal.beginEdit() = v;
    planeNormal.endEdit();
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
	//if (!vparams->isSupported(core::visual::API_OpenGL)) return;
    if (bDraw.getValue()) drawPlane(vparams);
}


template<class DataTypes>
void PlaneForceField<DataTypes>::drawPlane(const core::visual::VisualParams* vparams,float size)
{
    if (size == 0.0f) size = (float)drawSize.getValue();

    helper::ReadAccessor<VecCoord> p1 = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    defaulttype::Vec3d normal; normal = planeNormal.getValue();

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d(-normal[1]/normal[0], 1.0, 0.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, -normal[0]/normal[1],0.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 0.0, -normal[0]/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();

    defaulttype::Vec3d center = normal*planeD.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    std::vector< defaulttype::Vector3 > points;

    points.push_back(corners[0]);
    points.push_back(corners[1]);
    points.push_back(corners[2]);

    points.push_back(corners[0]);
    points.push_back(corners[2]);
    points.push_back(corners[3]);

    vparams->drawTool()->setPolygonMode(2,false); //Cull Front face

    vparams->drawTool()->drawTriangles(points, defaulttype::Vec<4,float>(color.getValue()[0],color.getValue()[1],color.getValue()[2],0.5));
    vparams->drawTool()->setPolygonMode(0,false); //No Culling

    std::vector< defaulttype::Vector3 > pointsLine;
    // lines for points penetrating the plane

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

	defaulttype::Vector3 point1,point2;
    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = DataTypes::getCPos(p1[i])*planeNormal.getValue()-planeD.getValue();
        CPos p2 = DataTypes::getCPos(p1[i]);
        p2 += planeNormal.getValue()*(-d);
        if (d<0)
        {
            point1 = DataTypes::getCPos(p1[i]);
            point2 = p2;
			pointsLine.push_back(point1);
			pointsLine.push_back(point2);
        }
    }
    vparams->drawTool()->drawLines(pointsLine, 1, defaulttype::Vec<4,float>(1,0,0,1));
}

template <class DataTypes>
void PlaneForceField<DataTypes>::computeBBox(const core::ExecParams * params, bool onlyVisible)
{
    if (onlyVisible && !bDraw.getValue()) return;

    const Real max_real = std::numeric_limits<Real>::max();
    const Real min_real = std::numeric_limits<Real>::min();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};

    defaulttype::Vec3d normal; normal = planeNormal.getValue(params);
    SReal size=10.0;

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d(-normal[1]/normal[0], 1.0, 0.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, -normal[0]/normal[1],0.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 0.0, -normal[0]/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();

    defaulttype::Vec3d center = normal*planeD.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    for (unsigned int i=0; i<4; i++)
    {
        for (int c=0; c<3; c++)
        {
            if (corners[i][c] > maxBBox[c]) maxBBox[c] = (Real)corners[i][c];
            if (corners[i][c] < minBBox[c]) minBBox[c] = (Real)corners[i][c];
        }
    }
    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
