/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL

#include "ConstantForceField.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
//#include <sofa/helper/gl/BasicShapes.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/behavior/ForceField.inl>



namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
ConstantForceField<DataTypes>::ConstantForceField()
    : points(initData(&points, "points", "points where the forces are applied"))
    , forces(initData(&forces, "forces", "applied forces at each point"))
    , force(initData(&force, "force", "applied force to all points if forces attribute is not specified"))
    , totalForce(initData(&totalForce, "totalForce", "total force for all points, will be distributed uniformly over points"))
    , arrowSizeCoef(initData(&arrowSizeCoef,0.0, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
    , indexFromEnd(initData(&indexFromEnd,(bool)false,"indexFromEnd", "Concerned DOFs indices are numbered from the end of the MState DOFs vector"))
{
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addForce(const core::MechanicalParams* /*params*/ /* PARAMS FIRST */, DataVecDeriv& f1, const DataVecCoord& p1, const DataVecDeriv&)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > _f1 = f1;
    _f1.resize(p1.getValue().size());

//        sout << "Points = " << points.getValue() << sendl;
    Deriv singleForce;
    if ( totalForce.getValue()*totalForce.getValue() > 0.0)
    {
        for (unsigned comp = 0; comp < totalForce.getValue().size(); comp++)
            singleForce[comp] = (totalForce.getValue()[comp])/(Real(points.getValue().size()));
        //std::cout << "Setting forces for each node to = " << singleForce << std::endl;
    }
    else if (force.getValue()*force.getValue() > 0.0)
    {
        singleForce = force.getValue();
        //std::cout << "Setting forces for each node to = " << singleForce << std::endl;
    }

    const VecIndex& indices = points.getValue();
//        sout << "indices = " << indices << sendl;
    const VecDeriv& f = forces.getValue();
    //const Deriv f_end = (f.empty()? force.getValue() : f[f.size()-1]);
    const Deriv f_end = (f.empty()? singleForce : f[f.size()-1]);
    unsigned int i = 0;


    if (!indexFromEnd.getValue())
    {
        for (; i < f.size(); i++)
        {
            sout<<"_f1[indices[i]] += f[i], "<< _f1[indices.empty() ? i : indices[i]] << " += " << f[i] << sendl;
            _f1[ indices.empty() ? i : indices[i] ] += f[i];  // if indices are not set, use the force indices

        }
        for (; i < indices.size(); i++)
        {
//                    sout<<"_f1[indices[i]] += f_end, "<< _f1[indices[i]] << " += " << f_end << sendl;
            _f1[indices[i]] += f_end;
        }
    }
    else
    {
        for (; i < f.size(); i++)
        {
            _f1[_f1.size() - indices[i] - 1] += f[i];
        }
        for (; i < indices.size(); i++)
        {
            _f1[_f1.size() - indices[i] - 1] += f_end;
        }
    }
}

template<class DataTypes>
void ConstantForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /* mat */, SReal /* k */, unsigned int & /* offset */)
{
}

template <class DataTypes>
double ConstantForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*params*/ /* PARAMS FIRST */, const DataVecCoord& x) const
{
    const VecIndex& indices = points.getValue();
    const VecDeriv& f = forces.getValue();
    const VecCoord& _x = x.getValue();
    const Deriv f_end = (f.empty()? force.getValue() : f[f.size()-1]);
    double e = 0;
    unsigned int i = 0;

    if (!indexFromEnd.getValue())
    {
        for (; i<f.size(); i++)
        {
            e -= f[i] * _x[indices[i]];
        }
        for (; i<indices.size(); i++)
        {
            e -= f_end * _x[indices[i]];
        }
    }
    else
    {
        for (; i < f.size(); i++)
        {
            e -= f[i] * _x[_x.size() - indices[i] -1];
        }
        for (; i < indices.size(); i++)
        {
            e -= f_end * _x[_x.size() - indices[i] -1];
        }
    }

    return e;
}


template <class DataTypes>
void ConstantForceField<DataTypes>::setForce(unsigned i, const Deriv& force)
{
    VecIndex& indices = *points.beginEdit();
    VecDeriv& f = *forces.beginEdit();
    indices.push_back(i);
    f.push_back( force );
    points.endEdit();
    forces.endEdit();
}


#ifndef SOFA_FLOAT
template <>
double ConstantForceField<defaulttype::Rigid3dTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& ) const;
template <>
double ConstantForceField<defaulttype::Rigid2dTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& ) const;
#endif

#ifndef SOFA_DOUBLE
template <>
double ConstantForceField<defaulttype::Rigid3fTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& ) const;
template <>
double ConstantForceField<defaulttype::Rigid2fTypes>::getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& ) const;
#endif


template<class DataTypes>
void ConstantForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    double aSC = arrowSizeCoef.getValue();

    Deriv singleForce;
    if (totalForce.getValue()*totalForce.getValue() > 0.0)
    {
        for (unsigned comp = 0; comp < totalForce.getValue().size(); comp++)
            singleForce[comp] = (totalForce.getValue()[comp])/(Real(points.getValue().size()));
    }
    else if (force.getValue() * force.getValue() > 0.0)
    {
        singleForce = force.getValue();
    }

    if ((!vparams->displayFlags().getShowForceFields() && (aSC==0)) || (aSC < 0.0)) return;
    const VecIndex& indices = points.getValue();
    const VecDeriv& f = forces.getValue();
    const Deriv f_end = (f.empty()? singleForce : f[f.size()-1]);
    const VecCoord& x = *this->mstate->getX();



    if( fabs(aSC)<1.0e-10 )
    {
        std::vector<defaulttype::Vector3> points;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Real xx,xy,xz,fx,fy,fz;

            if (!indexFromEnd.getValue())
            {
                DataTypes::get(xx,xy,xz,x[indices[i]]);
            }
            else
            {
                DataTypes::get(xx,xy,xz,x[x.size() - indices[i] - 1]);
            }

            DataTypes::get(fx,fy,fz,(i<f.size())? f[i] : f_end);
            points.push_back(defaulttype::Vector3(xx, xy, xz ));
            points.push_back(defaulttype::Vector3(xx+fx, xy+fy, xz+fz ));
        }
        vparams->drawTool()->drawLines(points, 2, defaulttype::Vec<4,float>(0,1,0,1));
    }
    else
    {
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Real xx,xy,xz,fx,fy,fz;

            if (!indexFromEnd.getValue())
            {
                DataTypes::get(xx,xy,xz,x[indices[i]]);
            }
            else
            {
                DataTypes::get(xx,xy,xz,x[x.size() - indices[i] - 1]);
            }

            DataTypes::get(fx,fy,fz,(i<f.size())? f[i] : f_end);

            defaulttype::Vector3 p1( xx, xy, xz);
            defaulttype::Vector3 p2( aSC*fx+xx, aSC*fy+xy, aSC*fz+xz );

            float norm = (float)(p2-p1).norm();

            if( aSC > 0)
            {
                //helper::gl::drawArrow(p1,p2, norm/20.0);
                vparams->drawTool()->drawArrow(p1,p2, norm/20.0f, defaulttype::Vec<4,float>(1.0f,0.4f,0.4f,1.0f));
            }
            else
            {
                //helper::gl::drawArrow(p2,p1, norm/20.0);
                vparams->drawTool()->drawArrow(p2,p1, norm/20.0f, defaulttype::Vec<4,float>(1.0f,0.4f,0.4f,1.0f));
            }
        }
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL



