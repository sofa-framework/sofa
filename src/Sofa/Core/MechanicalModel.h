#ifndef SOFA_CORE_MECHANICALMODEL_H
#define SOFA_CORE_MECHANICALMODEL_H

#include "BasicMechanicalModel.h"

namespace Sofa
{

namespace Core
{

template<class TDataTypes>
class MechanicalModel : public BasicMechanicalModel
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;

    virtual ~MechanicalModel() { }

    virtual VecCoord* getX() = 0;
    virtual VecDeriv* getV() = 0;
    virtual VecDeriv* getF() = 0;
    virtual VecDeriv* getDx() = 0;
    virtual VecConst* getC() = 0;

    virtual const VecCoord* getX()  const = 0;
    virtual const VecDeriv* getV()  const = 0;
    virtual const VecDeriv* getF()  const = 0;
    virtual const VecDeriv* getDx() const = 0;
    virtual const VecConst* getC() const = 0;

    virtual const VecCoord* getX0()  const = 0;
    virtual const VecDeriv* getV0()  const = 0;

    /// Get the indices of the particles located in the given bounding box
    virtual void getIndicesInSpace(std::vector<unsigned>& /*indices*/, Real /*xmin*/, Real /*xmax*/,Real /*ymin*/, Real /*ymax*/, Real /*zmin*/, Real /*zmax*/) const=0;
};

} // namespace Core

} // namespace Sofa

#endif
