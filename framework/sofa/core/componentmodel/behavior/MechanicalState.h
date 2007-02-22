#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MECHANICALSTATE_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_MECHANICALSTATE_H

#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class TDataTypes>
class MechanicalState : public BaseMechanicalState
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::SparseDeriv SparseDeriv;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::VecConst VecConst;

    virtual ~MechanicalState() { }

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


    /// Get the indices of the particles located in the given bounding box
    virtual void getIndicesInSpace(std::vector<unsigned>& /*indices*/, Real /*xmin*/, Real /*xmax*/,Real /*ymin*/, Real /*ymax*/, Real /*zmin*/, Real /*zmax*/) const=0;

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MechanicalState<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
