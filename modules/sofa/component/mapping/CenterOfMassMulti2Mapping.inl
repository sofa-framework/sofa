#include <sofa/component/mapping/CenterOfMassMulti2Mapping.h>
#include <sofa/defaulttype/Vec.h>
//#include <sofa/helper/gl/template.h>
#include <sofa/simulation/common/Simulation.h>
#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace mapping
{

using namespace core;
using namespace core::behavior;

template < typename Model >
struct Operation
{

    typedef typename Model::VecCoord VecCoord;
    typedef typename Model::Coord    Coord;
    typedef typename Model::Deriv    Deriv;
    typedef typename Model::VecDeriv VecDeriv;

public :
    static inline const VecCoord* getVecCoord( const Model* m, const VecId& id) { return m->getVecCoord(id.index); };
    static inline VecDeriv* getVecDeriv( Model* m, const VecId& id) { return m->getVecDeriv(id.index);};

    static inline const BaseMass* fetchMass  ( const Model* m)
    {
        BaseMass* mass = dynamic_cast<BaseMass*> (m->getContext()->getMass());
        return mass;
    }
    static inline double computeTotalMass( const Model* model, const BaseMass* mass )
    {
        double result = 0.0;
        for ( unsigned int i = 0; i < model->getX()->size(); i++)
        {
            result += mass->getElementMass(i);
        }
        return result;
    }

    static inline Coord WeightedCoord( const VecCoord* v, const BaseMass* m)
    {
        Coord c;
        for (unsigned int i=0 ; i< v->size() ; i++)
        {
            c += (*v)[i] * m->getElementMass(i);
        }
        return c;
    }

    static inline Deriv WeightedDeriv( const VecDeriv* v, const BaseMass* m)
    {
        Deriv d;
        for (unsigned int i=0 ; i< v->size() ; i++)
        {
            d += (*v)[i] * m->getElementMass(i);
        }
        return d;
    }
};

template< class BasicMulti2Mapping  >
void CenterOfMassMulti2Mapping< BasicMulti2Mapping >::apply(const helper::vector<typename Out::VecCoord*>& OutPos, const helper::vector<const typename In1::VecCoord*>& InPos1 , const helper::vector<const typename In2::VecCoord*>& InPos2 )
{
    assert( OutPos.size() == 1); // we are dealing with a many to one mapping.
    typedef typename helper::vector<typename In1::Coord>::iterator iter_coord1;
    typedef typename helper::vector<typename In2::Coord>::iterator iter_coord2;

    SReal px=0,py=0,pz=0;

    {
        In1Coord COM;
        std::transform(InPos1.begin(), InPos1.end(), inputBaseMass1.begin(), inputWeightedCOM1.begin(), Operation<In1>::WeightedCoord );

        for( iter_coord1 iter = inputWeightedCOM1.begin() ; iter != inputWeightedCOM1.end(); ++iter ) COM += *iter;
        COM *= invTotalMass;

        SReal x,y,z;
        In1::DataTypes::get(x,y,z,COM);
        px += x;
        py += y;
        pz += z;
    }

    {
        In2Coord COM;
        std::transform(InPos2.begin(), InPos2.end(), inputBaseMass2.begin(), inputWeightedCOM2.begin(), Operation<In2>::WeightedCoord );

        for( iter_coord2 iter = inputWeightedCOM2.begin() ; iter != inputWeightedCOM2.end(); ++iter ) COM += *iter;
        COM *= invTotalMass;

        SReal x,y,z;
        In2::DataTypes::get(x,y,z,COM);
        px += x;
        py += y;
        pz += z;
    }

    typename Out::VecCoord* outVecCoord = OutPos[0];

    Out::DataTypes::set((*outVecCoord)[0], px,py,pz);
}

template< class BasicMulti2Mapping >
void CenterOfMassMulti2Mapping< BasicMulti2Mapping >::applyJ(const helper::vector< typename Out::VecDeriv*>& OutDeriv, const helper::vector<const typename In1::VecDeriv*>& InDeriv1, const helper::vector<const typename In2::VecDeriv*>& InDeriv2)
{
    assert( OutDeriv.size() == 1 );
    typedef typename helper::vector<typename In1::Deriv>::iterator                     iter_deriv1;
    typedef typename helper::vector<typename In2::Deriv>::iterator                     iter_deriv2;

    SReal px=0,py=0,pz=0;

    {
        typename In1::Deriv Velocity;
        std::transform(InDeriv1.begin(), InDeriv1.end(), inputBaseMass1.begin(), inputWeightedForce1.begin(), Operation<In1>::WeightedDeriv );

        for ( iter_deriv1 iter = inputWeightedForce1.begin() ; iter != inputWeightedForce1.end() ; ++iter ) Velocity += *iter;
        Velocity *= invTotalMass;

        SReal x,y,z;
        In1::DataTypes::get(x,y,z,Velocity);
        px += x;
        py += y;
        pz += z;
    }

    {
        typename In2::Deriv Velocity;
        std::transform(InDeriv2.begin(), InDeriv2.end(), inputBaseMass2.begin(), inputWeightedForce2.begin(), Operation<In2>::WeightedDeriv );

        for ( iter_deriv2 iter = inputWeightedForce2.begin() ; iter != inputWeightedForce2.end() ; ++iter ) Velocity += *iter;
        Velocity *= invTotalMass;

        SReal x,y,z;
        In2::DataTypes::get(x,y,z,Velocity);
        px += x;
        py += y;
        pz += z;
    }

    typename Out::VecDeriv* outVecDeriv =  OutDeriv[0];

    Out::DataTypes::set((*outVecDeriv)[0], px,py,pz);
}



template < class BasicMulti2Mapping >
void CenterOfMassMulti2Mapping< BasicMulti2Mapping >::applyJT( const helper::vector<typename In1::VecDeriv*>& OutDeriv1 ,const helper::vector<typename In2::VecDeriv*>& OutDeriv2 , const helper::vector<const typename Out::VecDeriv*>& InDeriv )
{
    assert( InDeriv.size() == 1 );
    typedef helper::vector<const BaseMass*>::iterator iter_mass;


    typename Out::Deriv gravityCenterForce;
    const typename Out::VecDeriv* inForce = InDeriv[0];
    if( !inForce->empty() )
    {
        gravityCenterForce = (* inForce) [0];
        gravityCenterForce *= invTotalMass;

        SReal x,y,z;
        Out::DataTypes::get(x,y,z,gravityCenterForce);

        {
            typename In1::Deriv f;
            In1::DataTypes::set(f,x,y,z);

            for (unsigned int i=0; i<OutDeriv1.size(); ++i)
            {
                typename In1::VecDeriv& v=*(OutDeriv1[i]);
                const core::behavior::BaseMass* m=inputBaseMass1[i];
                for (unsigned int p=0; p<v.size(); ++p)
                {
                    v[p] += f*m->getElementMass(p);
                }
            }
        }

        {
            typename In2::Deriv f;
            In2::DataTypes::set(f,x,y,z);

            for (unsigned int i=0; i<OutDeriv2.size(); ++i)
            {
                typename In2::VecDeriv& v=*(OutDeriv2[i]);
                const core::behavior::BaseMass* m=inputBaseMass2[i];
                for (unsigned int p=0; p<v.size(); ++p)
                {
                    v[p] += f*m->getElementMass(p);
                }
            }
        }
    }
}

template< class BasicMulti2Mapping >
void CenterOfMassMulti2Mapping< BasicMulti2Mapping>::init()
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

    std::transform(this->fromModels1.begin(), this->fromModels1.end(), inputBaseMass1.begin(), Operation<In1>::fetchMass );
    std::transform(this->fromModels2.begin(), this->fromModels2.end(), inputBaseMass2.begin(), Operation<In2>::fetchMass );
    std::transform(this->fromModels1.begin(), this->fromModels1.end(), inputBaseMass1.begin(), inputTotalMass1.begin(), Operation<In1>::computeTotalMass );
    std::transform(this->fromModels2.begin(), this->fromModels2.end(), inputBaseMass2.begin(), inputTotalMass2.begin(), Operation<In2>::computeTotalMass );
    invTotalMass = 0.0;
    for ( iter_double iter = inputTotalMass1.begin() ; iter != inputTotalMass1.end() ; ++ iter ) invTotalMass += *iter;
    for ( iter_double iter = inputTotalMass2.begin() ; iter != inputTotalMass2.end() ; ++ iter ) invTotalMass += *iter;
    invTotalMass = 1.0/invTotalMass;
    Inherit::init();

    if (this->getToModels()[0]) this->getToModels()[0]->resize(1);
}

template< class BasicMulti2Mapping >
void CenterOfMassMulti2Mapping< BasicMulti2Mapping >::draw()
{
    assert( this->toModels.size() == 1 );
    OutVecCoord* X = this->getToModels()[0]->getVecCoord( VecId::position().index );

    std::vector< Vector3 > points;
    Vector3 point1,point2;

    unsigned int sizePoints= (OutCoord::static_size <=3)?OutCoord::static_size:3;
    for(unsigned int i=0 ; i<OutCoord::static_size ; i++)
    {
        OutCoord v;
        v[i] = (Real)0.1;
        for (unsigned int s=0; s<sizePoints; ++s)
        {
            point1[s] = ( (*X)[0] -v)[s];
            point2[s] = ( (*X)[0] +v)[s];
        }
        points.push_back(point1);
        points.push_back(point2);
    }
    simulation::getSimulation()->DrawUtility.drawLines(points, 1, Vec<4,float>(1,1,0,1));

}


}
}
}
