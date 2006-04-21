#ifndef SOFA_CORE_MECHANICALMODEL_H
#define SOFA_CORE_MECHANICALMODEL_H

#include "BasicMechanicalModel.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
class MechanicalModel : public BasicMechanicalModel
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    virtual ~MechanicalModel() { }

    virtual VecCoord* getX() = 0;
    virtual VecDeriv* getV() = 0;
    virtual VecDeriv* getF() = 0;
    virtual VecDeriv* getDx() = 0;

    virtual const VecCoord* getX()  const = 0;
    virtual const VecDeriv* getV()  const = 0;
    virtual const VecDeriv* getF()  const = 0;
    virtual const VecDeriv* getDx() const = 0;
};

} // namespace Core

} // namespace Sofa

#endif
