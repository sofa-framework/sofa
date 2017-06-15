/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_INL
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_INL

#include <SofaMiscMapping/CenterOfMassMulti2Mapping.h>
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

template< class TIn1, class TIn2, class TOut  >
void CenterOfMassMulti2Mapping< TIn1, TIn2, TOut >::apply(
        const core::MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos,
        const helper::vector<const In1DataVecCoord*>& dataVecIn1Pos ,
        const helper::vector<const In2DataVecCoord*>& dataVecIn2Pos)
//apply(const vecOutVecCoord& outPos, const vecConstIn1VecCoord& inPos1 , const vecConstIn2VecCoord& inPos2 )
{
    assert( dataVecOutPos.size() == 1); // we are dealing with a many to one mapping.
    typedef typename helper::vector<In1Coord>::iterator iter_coord1;
    typedef typename helper::vector<In2Coord>::iterator iter_coord2;

    SReal px=0,py=0,pz=0;

    //Not optimized at all...
    helper::vector<OutVecCoord*> outPos;
    for(unsigned int i=0; i<dataVecOutPos.size(); i++)
        outPos.push_back(dataVecOutPos[i]->beginEdit(mparams));

    helper::vector<const In1VecCoord*> inPos1;
    for(unsigned int i=0; i<dataVecIn1Pos.size(); i++)
        inPos1.push_back(&dataVecIn1Pos[i]->getValue(mparams));
    helper::vector<const In2VecCoord*> inPos2;
    for(unsigned int i=0; i<dataVecIn2Pos.size(); i++)
        inPos2.push_back(&dataVecIn2Pos[i]->getValue(mparams));


    {
        In1Coord COM;
        std::transform(inPos1.begin(), inPos1.end(), inputBaseMass1.begin(), inputWeightedCOM1.begin(), Operation< core::State<In1> >::WeightedCoord );

        for( iter_coord1 iter = inputWeightedCOM1.begin() ; iter != inputWeightedCOM1.end(); ++iter ) COM += *iter;
        COM *= invTotalMass;

        SReal x,y,z;
        In1DataTypes::get(x,y,z,COM);
        px += x;
        py += y;
        pz += z;
    }

    {
        In2Coord COM;
        std::transform(inPos2.begin(), inPos2.end(), inputBaseMass2.begin(), inputWeightedCOM2.begin(), Operation< core::State<In2> >::WeightedCoord );

        for( iter_coord2 iter = inputWeightedCOM2.begin() ; iter != inputWeightedCOM2.end(); ++iter ) COM += *iter;
        COM *= invTotalMass;

        SReal x,y,z;
        In2DataTypes::get(x,y,z,COM);
        px += x;
        py += y;
        pz += z;
    }

    OutVecCoord* outVecCoord = outPos[0];

    OutDataTypes::set((*outVecCoord)[0], px,py,pz);

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutPos.size(); i++)
        dataVecOutPos[i]->endEdit(mparams);
}

template <class TIn1, class TIn2, class TOut>
void CenterOfMassMulti2Mapping< TIn1, TIn2, TOut >::applyJ(
        const core::MechanicalParams* mparams, const helper::vector< OutDataVecDeriv*>& dataVecOutVel,
        const helper::vector<const In1DataVecDeriv*>& dataVecIn1Vel,
        const helper::vector<const In2DataVecDeriv*>& dataVecIn2Vel)
//applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const In1VecDeriv*>& inDeriv1, const helper::vector<const In2VecDeriv*>& inDeriv2)
{
    assert( dataVecOutVel.size() == 1 );
    typedef typename helper::vector<In1Deriv>::iterator                     iter_deriv1;
    typedef typename helper::vector<In2Deriv>::iterator                     iter_deriv2;

    //Not optimized at all...
    helper::vector<OutVecDeriv*> outDeriv;
    for(unsigned int i=0; i<dataVecOutVel.size(); i++)
        outDeriv.push_back(dataVecOutVel[i]->beginEdit(mparams));

    helper::vector<const In1VecDeriv*> inDeriv1;
    for(unsigned int i=0; i<dataVecIn1Vel.size(); i++)
        inDeriv1.push_back(&dataVecIn1Vel[i]->getValue(mparams));
    helper::vector<const In2VecDeriv*> inDeriv2;
    for(unsigned int i=0; i<dataVecIn2Vel.size(); i++)
        inDeriv2.push_back(&dataVecIn2Vel[i]->getValue(mparams));

    SReal px=0,py=0,pz=0;

    {
        In1Deriv Velocity;
        std::transform(inDeriv1.begin(), inDeriv1.end(), inputBaseMass1.begin(), inputWeightedForce1.begin(), Operation< core::State<In1> >::WeightedDeriv );

        for ( iter_deriv1 iter = inputWeightedForce1.begin() ; iter != inputWeightedForce1.end() ; ++iter ) Velocity += *iter;
        Velocity *= invTotalMass;

        SReal x,y,z;
        In1DataTypes::get(x,y,z,Velocity);
        px += x;
        py += y;
        pz += z;
    }

    {
        In2Deriv Velocity;
        std::transform(inDeriv2.begin(), inDeriv2.end(), inputBaseMass2.begin(), inputWeightedForce2.begin(), Operation< core::State<In2> >::WeightedDeriv );

        for ( iter_deriv2 iter = inputWeightedForce2.begin() ; iter != inputWeightedForce2.end() ; ++iter ) Velocity += *iter;
        Velocity *= invTotalMass;

        SReal x,y,z;
        In2DataTypes::get(x,y,z,Velocity);
        px += x;
        py += y;
        pz += z;
    }

    OutVecDeriv* outVecDeriv =  outDeriv[0];

    OutDataTypes::set((*outVecDeriv)[0], px,py,pz);

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOutVel.size(); i++)
        dataVecOutVel[i]->endEdit(mparams);
}



template < class TIn1, class TIn2, class TOut >
void CenterOfMassMulti2Mapping< TIn1, TIn2, TOut >::applyJT(
        const core::MechanicalParams* mparams, const helper::vector< In1DataVecDeriv*>& dataVecOut1Force,
        const helper::vector< In2DataVecDeriv*>& dataVecOut2Force,
        const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
//applyJT( const helper::vector<typename In1::VecDeriv*>& outDeriv1 ,const helper::vector<typename In2::VecDeriv*>& outDeriv2 , const helper::vector<const typename Out::VecDeriv*>& inDeriv )
{
    assert( dataVecOut1Force.size() == 1 );

    //Not optimized at all...
    helper::vector<In1VecDeriv*> outDeriv1;
    for(unsigned int i=0; i<dataVecOut1Force.size(); i++)
        outDeriv1.push_back(dataVecOut1Force[i]->beginEdit(mparams));
    helper::vector<In2VecDeriv*> outDeriv2;
    for(unsigned int i=0; i<dataVecOut2Force.size(); i++)
        outDeriv2.push_back(dataVecOut2Force[i]->beginEdit(mparams));

    helper::vector<const OutVecDeriv*> inDeriv;
    for(unsigned int i=0; i<dataVecInForce.size(); i++)
        inDeriv.push_back(&dataVecInForce[i]->getValue(mparams));

    typename Out::Deriv gravityCenterForce;
    const typename Out::VecDeriv* inForce = inDeriv[0];
    if( !inForce->empty() )
    {
        gravityCenterForce = (* inForce) [0];
        gravityCenterForce *= invTotalMass;

        SReal x,y,z;
        OutDataTypes::get(x,y,z,gravityCenterForce);

        {
            typename In1::Deriv f;
            In1DataTypes::set(f,x,y,z);

            for (unsigned int i=0; i<outDeriv1.size(); ++i)
            {
                typename In1::VecDeriv& v=*(outDeriv1[i]);
                const core::behavior::BaseMass* m=inputBaseMass1[i];
                for (unsigned int p=0; p<v.size(); ++p)
                {
                    v[p] += f*m->getElementMass(p);
                }
            }
        }

        {
            typename In2::Deriv f;
            In2DataTypes::set(f,x,y,z);

            for (unsigned int i=0; i<outDeriv2.size(); ++i)
            {
                typename In2::VecDeriv& v=*(outDeriv2[i]);
                const core::behavior::BaseMass* m=inputBaseMass2[i];
                for (unsigned int p=0; p<v.size(); ++p)
                {
                    v[p] += f*m->getElementMass(p);
                }
            }
        }
    }

    //Really Not optimized at all...
    for(unsigned int i=0; i<dataVecOut1Force.size(); i++)
    {
        dataVecOut1Force[i]->endEdit(mparams);
    }
    for(unsigned int i=0; i<dataVecOut2Force.size(); i++)
    {
        dataVecOut2Force[i]->endEdit(mparams);
    }
}

template <class TIn1, class TIn2, class TOut>
void CenterOfMassMulti2Mapping< TIn1, TIn2, TOut>::init()
{
    typedef helper::vector<double>::iterator  iter_double;

    inputBaseMass1.resize ( this->fromModels1.size()  );
    inputTotalMass1.resize( this->fromModels1.size()  );
    inputWeightedCOM1  .resize( this->fromModels1.size() );
    inputWeightedForce1.resize( this->fromModels1.size() );

    inputBaseMass2.resize ( this->fromModels2.size()  );
    inputTotalMass2.resize( this->fromModels2.size()  );
    inputWeightedCOM2  .resize( this->fromModels2.size() );
    inputWeightedForce2.resize( this->fromModels2.size() );

    std::transform(this->fromModels1.begin(), this->fromModels1.end(), inputBaseMass1.begin(), Operation< core::State<In1> >::fetchMass );
    std::transform(this->fromModels2.begin(), this->fromModels2.end(), inputBaseMass2.begin(), Operation< core::State<In2> >::fetchMass );
    std::transform(this->fromModels1.begin(), this->fromModels1.end(), inputBaseMass1.begin(), inputTotalMass1.begin(), Operation< core::State<In1> >::computeTotalMass );
    std::transform(this->fromModels2.begin(), this->fromModels2.end(), inputBaseMass2.begin(), inputTotalMass2.begin(), Operation< core::State<In2> >::computeTotalMass );
    invTotalMass = 0.0;
    for ( iter_double iter = inputTotalMass1.begin() ; iter != inputTotalMass1.end() ; ++ iter ) invTotalMass += *iter;
    for ( iter_double iter = inputTotalMass2.begin() ; iter != inputTotalMass2.end() ; ++ iter ) invTotalMass += *iter;
    invTotalMass = 1.0/invTotalMass;
    Inherit::init();

    if (this->getToModels()[0]) this->getToModels()[0]->resize(1);
}

template <class TIn1, class TIn2, class TOut>
void CenterOfMassMulti2Mapping< TIn1, TIn2, TOut >::draw(const core::visual::VisualParams* vparams)
{
    assert( this->toModels.size() == 1 );
    const Data< OutVecCoord > *X = this->getToModels()[0]->read(sofa::core::VecCoordId::position());

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

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_INL
