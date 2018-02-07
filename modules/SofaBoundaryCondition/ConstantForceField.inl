/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL

#include <SofaBoundaryCondition/ConstantForceField.h>
#include <sofa/helper/system/config.h>
#include <assert.h>
#include <iostream>
#include <sofa/simulation/Simulation.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TopologySubsetData.inl>

#include <sofa/core/objectmodel/BaseObjectDescription.h>
using sofa::core::objectmodel::BaseObjectDescription ;

namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
ConstantForceField<DataTypes>::ConstantForceField()
    : d_indices(initData(&d_indices, "indices",
                         "indices where the forces are applied"))

    , d_indexFromEnd(initData(&d_indexFromEnd,(bool)false,"indexFromEnd",
                              "Concerned DOFs indices are numbered from the end of the MState DOFs vector. (default=false)"))

    , d_forces(initData(&d_forces, "forces",
                        "applied forces at each point"))

    , d_force(initData(&d_force, "force",
                       "applied force to all points if forces attribute is not specified"))

    , d_totalForce(initData(&d_totalForce, "totalForce",
                            "total force for all points, will be distributed uniformly over points"))

    , d_arrowSizeCoef(initData(&d_arrowSizeCoef,(SReal)0.0, "arrowSizeCoef",
                               "Size of the drawn arrows (0->no arrows, sign->direction of drawing. (default=0)"))

    , d_color(initData(&d_color, defaulttype::RGBAColor(0.2f,0.9f,0.3f,1.0f), "showColor",
                       "Color for object display (default: [0.2,0.9,0.3,1.0])"))
{
    d_arrowSizeCoef.setGroup("Visualization");
    d_color.setGroup("Visualization");

    /// We add aliases only for deprecated attributes and as long as we are handling them.
    addAlias(&d_indices, "points") ;
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams,
                                              DataVecDeriv& d_df , const DataVecDeriv& d_dx)
{
    //TODO: remove this line (avoid warning message) ...
    mparams->setKFactorUsed(true);
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > _f1 = d_df;
    _f1.resize(d_dx.getValue().size());
}

template<class DataTypes>
void ConstantForceField<DataTypes>::parse(BaseObjectDescription* arg)
{
    /// Now handling backward compatibility with old scenes.
    /// point is deprecated since '17.06'
    const char* val=arg->getAttribute("points",nullptr) ;
    if(val)
    {
        msg_deprecated() << "The attribute 'points' is deprecated since Sofa 17.06 and converted into 'indices'." << msgendl
                         << "Using deprecated attribute may result in lower performance and undefined behavior." << msgendl
                         << "To remove this message you need to replace the 'points' attributes with the 'indices' one. ";

    }
    Inherit::parse(arg) ;
}

template<class DataTypes>
void ConstantForceField<DataTypes>::init()
{
    m_topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters for topology data and handler
    d_indices.createTopologicalEngine(m_topology);
    d_indices.registerTopologicalData();

    Inherit::init();
}


template<class DataTypes>
void ConstantForceField<DataTypes>::addForce(const core::MechanicalParams* /*params*/,
                                             DataVecDeriv& f1, const DataVecCoord& p1, const DataVecDeriv&)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > _f1 = f1;
    _f1.resize(p1.getValue().size());

    Deriv singleForce;
    const Deriv& forceVal = d_force.getValue();
    const Deriv& totalForceVal = d_totalForce.getValue();
    const VecIndex& indices = d_indices.getValue();
    const VecDeriv& f = d_forces.getValue();
    unsigned int i = 0, nbForcesIn = f.size(), nbForcesOut = _f1.size();

    if (totalForceVal * totalForceVal > 0)
    {
        unsigned int nbForces = indices.empty() ? nbForcesOut : indices.size();
        singleForce = totalForceVal / (Real)nbForces;
    }
    else if (forceVal * forceVal > 0.0)
        singleForce = forceVal;

    const Deriv f_end = (f.empty() ? singleForce : f[f.size()-1]);

    // When no indices are given, copy the forces from the start
    if (indices.empty())
    {
        unsigned int nbCopy = std::min(nbForcesIn, nbForcesOut);
        for (; i < nbCopy; ++i) // Copy from the forces list
            _f1[i] += f[i];
        for (; i < nbForcesOut; ++i) // Copy from the single value or the last value of the forces list
            _f1[i] += f_end;
    }
    else
    {
        unsigned int nbIndices = indices.size();
        unsigned int nbCopy = std::min(nbForcesIn, nbIndices); // forces & points are not garanteed to be of the same size
        if (!d_indexFromEnd.getValue())
        {
            for (; i < nbCopy; ++i)
                _f1[indices[i]] += f[i];
            for (; i < nbIndices; ++i)
                _f1[indices[i]] += f_end;
        }
        else
        {
            for (; i < nbCopy; ++i)
                _f1[nbForcesOut - indices[i] - 1] += f[i];
            for (; i < nbIndices; ++i)
                _f1[nbForcesOut - indices[i] - 1] += f_end;
        }
    }
}

template<class DataTypes>
void ConstantForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /* mat */,
                                                 SReal /* k */, unsigned int & /* offset */)
{
}

template <class DataTypes>
SReal ConstantForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*params*/,
                                                        const DataVecCoord& x) const
{
    const VecIndex& indices = d_indices.getValue();
    const VecDeriv& f = d_forces.getValue();
    const VecCoord& _x = x.getValue();
    const Deriv f_end = (f.empty()? d_force.getValue() : f[f.size()-1]);
    SReal e = 0;
    unsigned int i = 0;

    if (!d_indexFromEnd.getValue())
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
    VecIndex& indices = *d_indices.beginEdit();
    VecDeriv& f = *d_forces.beginEdit();
    indices.push_back(i);
    f.push_back( force );
    d_indices.endEdit();
    d_forces.endEdit();
}

template<class DataTypes>
void ConstantForceField<DataTypes>::addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/,
                                                 SReal /*kFact*/)
{}


template<class DataTypes>
void ConstantForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    vparams->drawTool()->saveLastState();

    SReal aSC = d_arrowSizeCoef.getValue();

    Deriv singleForce;
    if (d_totalForce.getValue()*d_totalForce.getValue() > 0.0)
    {
        for (unsigned comp = 0; comp < d_totalForce.getValue().size(); comp++)
            singleForce[comp] = (d_totalForce.getValue()[comp])/(Real(d_indices.getValue().size()));
    }
    else if (d_force.getValue() * d_force.getValue() > 0.0)
    {
        singleForce = d_force.getValue();
    }

    if ((!vparams->displayFlags().getShowForceFields() && (aSC==0)) || (aSC < 0.0)) return;
    const VecIndex& indices = d_indices.getValue();
    const VecDeriv& f = d_forces.getValue();
    const Deriv f_end = (f.empty()? singleForce : f[f.size()-1]);
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();



    if( fabs(aSC)<1.0e-10 )
    {
        std::vector<defaulttype::Vector3> points;
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Real xx,xy,xz,fx,fy,fz;

            if (!d_indexFromEnd.getValue())
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

        vparams->drawTool()->setLightingEnabled(true);
        for (unsigned int i=0; i<indices.size(); i++)
        {
            Real xx,xy,xz,fx,fy,fz;

            if (!d_indexFromEnd.getValue())
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
                vparams->drawTool()->drawArrow(p1,p2, norm/20.0f, d_color.getValue());
            }
            else
            {
                vparams->drawTool()->drawArrow(p2,p1, norm/20.0f, d_color.getValue());
            }
        }
    }

    vparams->drawTool()->restoreLastState();
}

template<class DataTypes>
void ConstantForceField<DataTypes>::updateForceMask()
{
    const VecIndex& indices = d_indices.getValue();

    for (size_t i=0; i<indices.size(); i++)
    {
        this->mstate->forceMask.insertEntry(i);
    }
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_CONSTANTFORCEFIELD_INL



