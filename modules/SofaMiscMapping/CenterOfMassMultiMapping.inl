/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_INL
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_INL

#include <SofaMiscMapping/CenterOfMassMultiMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/Simulation.h>

#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace mapping
{

template < typename Model >
struct Operation
{
    typedef typename Model::VecCoord VecCoord;
    typedef typename Model::Coord    Coord;
    typedef typename Model::Deriv    Deriv;
    typedef typename Model::VecDeriv VecDeriv;

public :
    static inline const VecCoord* getVecCoord( const Model* m, const sofa::core::VecId id) { return m->getVecCoord(id.index); }
    static inline VecDeriv* getVecDeriv( Model* m, const sofa::core::VecId id) { return m->getVecDeriv(id.index);}

    static inline const sofa::core::behavior::BaseMass* fetchMass  ( const Model* m)
    {
        sofa::core::behavior::BaseMass* mass = m->getContext()->getMass();
        return mass;
    }
    static inline double computeTotalMass( const Model* model, const sofa::core::behavior::BaseMass* mass )
    {
        double result = 0.0;
        const unsigned int modelSize = static_cast<unsigned int>(model->getSize());
        for (unsigned int i = 0; i < modelSize; i++)
        {
            result += mass->getElementMass(i);
        }
        return result;
    }

    static inline Coord WeightedCoord( const VecCoord* v, const sofa::core::behavior::BaseMass* m)
    {
        Coord c;
        for (unsigned int i=0 ; i< v->size() ; i++)
        {
            c += (*v)[i] * m->getElementMass(i);
        }
        return c;
    }

    static inline Deriv WeightedDeriv( const VecDeriv* v, const sofa::core::behavior::BaseMass* m)
    {
        Deriv d;
        for (unsigned int i=0 ; i< v->size() ; i++)
        {
            d += (*v)[i] * m->getElementMass(i);
        }
        return d;
    }
};


template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut >::apply(const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
{
    typedef typename InVecCoord::iterator iter_coord;

    //Not optimized at all...
    helper::vector<OutVecCoord*> outPos;
    for(unsigned int i=0; i<dataVecOutPos.size(); i++)
        outPos.push_back(dataVecOutPos[i]->beginEdit(mparams));

    helper::vector<const InVecCoord*> inPos;
    for(unsigned int i=0; i<dataVecInPos.size(); i++)
        inPos.push_back(&dataVecInPos[i]->getValue(mparams));

    assert( outPos.size() == 1); // we are dealing with a many to one mapping.
    InCoord COM;
    std::transform(inPos.begin(), inPos.end(), inputBaseMass.begin(), inputWeightedCOM.begin(), Operation< core::State<In> >::WeightedCoord );

    for( iter_coord iter = inputWeightedCOM.begin() ; iter != inputWeightedCOM.end(); ++iter ) COM += *iter;
    COM *= invTotalMass;

    OutVecCoord* outVecCoord = outPos[0];

    SReal x,y,z;
    InDataTypes::get(x,y,z,COM);
    OutDataTypes::set((*outVecCoord)[0], x,y,z);

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutPos.size(); i++)
        dataVecOutPos[i]->endEdit(mparams);
}


template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut >::applyJ(const core::MechanicalParams* mparams, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
{
    typedef typename InVecDeriv::iterator iter_deriv;

    //Not optimized at all...
    helper::vector<OutVecDeriv*> outDeriv;
    for(unsigned int i=0; i<dataVecOutVel.size(); i++)
        outDeriv.push_back(dataVecOutVel[i]->beginEdit(mparams));

    helper::vector<const InVecDeriv*> inDeriv;
    for(unsigned int i=0; i<dataVecInVel.size(); i++)
        inDeriv.push_back(&dataVecInVel[i]->getValue(mparams));

    assert( outDeriv.size() == 1 );

    InDeriv Velocity;
    std::transform(inDeriv.begin(), inDeriv.end(), inputBaseMass.begin(), inputWeightedForce.begin(), Operation<In>::WeightedDeriv );

    for ( iter_deriv iter = inputWeightedForce.begin() ; iter != inputWeightedForce.end() ; ++iter ) Velocity += *iter;
    Velocity *= invTotalMass;

    OutVecDeriv* outVecDeriv =  outDeriv[0];

    SReal x,y,z;
    InDataTypes::get(x,y,z,Velocity);
    OutDataTypes::set((*outVecDeriv)[0], x,y,z);

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutVel.size(); i++)
        dataVecOutVel[i]->endEdit(mparams);
}


template < class TIn, class TOut >
void CenterOfMassMultiMapping< TIn, TOut >::applyJT(const core::MechanicalParams* mparams, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
{
    //Not optimized at all...
    helper::vector<InVecDeriv*> outDeriv;
    for(unsigned int i=0; i<dataVecOutForce.size(); i++)
        outDeriv.push_back(dataVecOutForce[i]->beginEdit(mparams));

    helper::vector<const OutVecDeriv*> inDeriv;
    for(unsigned int i=0; i<dataVecInForce.size(); i++)
        inDeriv.push_back(&dataVecInForce[i]->getValue(mparams));


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
        dataVecOutForce[i]->endEdit(mparams);
}


template <class TIn, class TOut>
void CenterOfMassMultiMapping< TIn, TOut>::init()
{
    typedef helper::vector<double>::iterator iter_double;

    inputBaseMass.resize ( this->getFromModels().size()  );
    inputTotalMass.resize( this->getFromModels().size()  );
    inputWeightedCOM.resize( this->getFromModels().size() );
    inputWeightedForce.resize( this->getFromModels().size() );

    std::transform(this->getFromModels().begin(), this->getFromModels().end(), inputBaseMass.begin(), Operation< core::State<In> >::fetchMass );

    std::transform(this->getFromModels().begin(), this->getFromModels().end(), inputBaseMass.begin(), inputTotalMass.begin(), Operation< core::State<In> >::computeTotalMass );

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

    std::vector< sofa::defaulttype::Vector3 > points;
    sofa::defaulttype::Vector3 point1,point2;

    for(unsigned int i=0 ; i<OutCoord::spatial_dimensions ; i++)
    {
        OutCoord v;
        v[i] = (Real)0.1;
        point1 = OutDataTypes::getCPos(X->getValue()[0] - v);
        point2 = OutDataTypes::getCPos(X->getValue()[0] + v);
        points.push_back(point1);
        points.push_back(point2);
    }

    vparams->drawTool()->drawLines(points, 1, sofa::defaulttype::Vec<4,float>(1,1,0,1));
}


} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_INL
