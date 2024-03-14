/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/mapping/linear/CenterOfMassMultiMapping.h>
#include <sofa/component/mapping/linear/CenterOfMassMappingOperation.h>

#include <sofa/core/visual/VisualParams.h>

#include <algorithm>
#include <functional>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut >::apply(const core::MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos)
{
    SOFA_UNUSED(mparams);

    typedef typename InVecCoord::iterator iter_coord;

    //Not optimized at all...
    type::vector<OutVecCoord*> outPos;
    for(unsigned int i=0; i<dataVecOutPos.size(); i++)
        outPos.push_back(dataVecOutPos[i]->beginEdit());

    type::vector<const InVecCoord*> inPos;
    for(unsigned int i=0; i<dataVecInPos.size(); i++)
        inPos.push_back(&dataVecInPos[i]->getValue());

    assert( outPos.size() == 1); // we are dealing with a many to one mapping.
    InCoord COM;
    std::transform(inPos.begin(), inPos.end(), inputBaseMass.begin(), inputWeightedCOM.begin(), CenterOfMassMappingOperation< core::State<In> >::WeightedCoord );

    for( iter_coord iter = inputWeightedCOM.begin() ; iter != inputWeightedCOM.end(); ++iter ) COM += *iter;
    COM *= invTotalMass;

    OutVecCoord* outVecCoord = outPos[0];

    SReal x,y,z;
    InDataTypes::get(x,y,z,COM);
    OutDataTypes::set((*outVecCoord)[0], x,y,z);

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutPos.size(); i++)
        dataVecOutPos[i]->endEdit();
}


template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut >::applyJ(const core::MechanicalParams* mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel)
{
    SOFA_UNUSED(mparams);

    typedef typename InVecDeriv::iterator iter_deriv;

    //Not optimized at all...
    type::vector<OutVecDeriv*> outDeriv;
    for(unsigned int i=0; i<dataVecOutVel.size(); i++)
        outDeriv.push_back(dataVecOutVel[i]->beginEdit());

    type::vector<const InVecDeriv*> inDeriv;
    for(unsigned int i=0; i<dataVecInVel.size(); i++)
        inDeriv.push_back(&dataVecInVel[i]->getValue());

    assert( outDeriv.size() == 1 );

    InDeriv Velocity;
    std::transform(inDeriv.begin(), inDeriv.end(), inputBaseMass.begin(), inputWeightedForce.begin(), CenterOfMassMappingOperation<In>::WeightedDeriv );

    for ( iter_deriv iter = inputWeightedForce.begin() ; iter != inputWeightedForce.end() ; ++iter ) Velocity += *iter;
    Velocity *= invTotalMass;

    OutVecDeriv* outVecDeriv =  outDeriv[0];

    SReal x,y,z;
    InDataTypes::get(x,y,z,Velocity);
    OutDataTypes::set((*outVecDeriv)[0], x,y,z);

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutVel.size(); i++)
        dataVecOutVel[i]->endEdit();
}


template < class TIn, class TOut >
void CenterOfMassMultiMapping< TIn, TOut >::applyJT(const core::MechanicalParams* mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    SOFA_UNUSED(mparams);

    //Not optimized at all...
    type::vector<InVecDeriv*> outDeriv;
    for(unsigned int i=0; i<dataVecOutForce.size(); i++)
        outDeriv.push_back(dataVecOutForce[i]->beginEdit());

    type::vector<const OutVecDeriv*> inDeriv;
    for(unsigned int i=0; i<dataVecInForce.size(); i++)
        inDeriv.push_back(&dataVecInForce[i]->getValue());


    assert( inDeriv.size() == 1 );

    OutDeriv gravityCenterForce;
    const OutVecDeriv* inForce = inDeriv[0];
    if( !inForce->empty() )
    {
        gravityCenterForce = (* inForce) [0];
        gravityCenterForce *= invTotalMass;

        SReal x,y,z;
        OutDataTypes::get(x,y,z,gravityCenterForce);

        InDeriv f;
        InDataTypes::set(f,x,y,z);

        for (unsigned int i=0; i<outDeriv.size(); ++i)
        {
            InVecDeriv& v=*(outDeriv[i]);
            const core::behavior::BaseMass* m=inputBaseMass[i];
            for (unsigned int p=0; p<v.size(); ++p)
            {
                v[p] += f*m->getElementMass(p);
            }
        }
    }

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutForce.size(); i++)
        dataVecOutForce[i]->endEdit();
}


template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut>::init()
{
    typedef type::vector<double>::iterator iter_double;

    inputBaseMass.resize ( this->getFromModels().size()  );
    inputTotalMass.resize( this->getFromModels().size()  );
    inputWeightedCOM.resize( this->getFromModels().size() );
    inputWeightedForce.resize( this->getFromModels().size() );

    std::transform(this->getFromModels().begin(), this->getFromModels().end(), inputBaseMass.begin(), CenterOfMassMappingOperation< core::State<In> >::fetchMass );

    std::transform(this->getFromModels().begin(), this->getFromModels().end(), inputBaseMass.begin(), inputTotalMass.begin(), CenterOfMassMappingOperation< core::State<In> >::computeTotalMass );

    invTotalMass = 0.0;
    for ( iter_double iter = inputTotalMass.begin() ; iter != inputTotalMass.end() ; ++ iter )
    {
        invTotalMass += *iter;
    }
    invTotalMass = 1.0/invTotalMass;
    Inherit::init();

    sofa::core::State<Out>* toModel = this->toModels[0];
    if (toModel) toModel->resize(1);
}


template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut >::draw(const core::visual::VisualParams* vparams)
{
    assert( this->toModels.size() == 1 );
    const sofa::core::objectmodel::Data< OutVecCoord > *X = this->getToModels()[0]->read(sofa::core::VecCoordId::position());

    std::vector< sofa::type::Vec3 > points;
    sofa::type::Vec3 point1,point2;

    for(unsigned int i=0 ; i<OutCoord::spatial_dimensions ; i++)
    {
        OutCoord v;
        v[i] = (Real)0.1;
        point1 = OutDataTypes::getCPos(X->getValue()[0] - v);
        point2 = OutDataTypes::getCPos(X->getValue()[0] + v);
        points.push_back(point1);
        points.push_back(point2);
    }

    vparams->drawTool()->drawLines(points, 1, sofa::type::RGBAColor::yellow());
}

} // namespace sofa::component::mapping::linear
