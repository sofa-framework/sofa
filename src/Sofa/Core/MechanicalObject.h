#ifndef SOFA_CORE_MECHANICALOBJECT_H
#define SOFA_CORE_MECHANICALOBJECT_H

#include "ForceField.h"
#include "BasicMechanicalMapping.h"
#include "MechanicalModel.h"
#include "BasicMechanicalObject.h"
#include "Mass.h"
#include <vector>
#include <assert.h>

namespace Sofa
{

namespace Core
{

template <class DataTypes>
class MechanicalObject : public MechanicalModel<DataTypes>, public BasicMechanicalObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    VecCoord* x;
    VecDeriv* v;
    VecDeriv* f;
    VecDeriv* dx;

    /// @name Integration-related data
    /// @{

    std::vector< VecCoord * > vectorsCoord;
    std::vector< VecDeriv * > vectorsDeriv;
    int vsize; ///< Number of elements to allocate in vectors

    void setVecCoord(unsigned int index, VecCoord* v);
    void setVecDeriv(unsigned int index, VecDeriv* v);

    /// @}

    /// @name Scene structure (only used if there is no external scenegraph)
    /// @{

    BasicMechanicalMapping* mapping;
    std::vector< Core::ForceField *> forcefields;
    std::vector< Core::Constraint *> constraints;
    std::vector< Core::BasicMechanicalObject *> mmodels;
    Topology* topology;
    Mass* mass;
    typedef typename std::vector< Core::ForceField* >::iterator ForceFieldIt;
    typedef typename std::vector< Core::Constraint* >::iterator ConstraintIt;
    typedef typename std::vector< Core::BasicMechanicalObject* >::iterator MModelIt;

    /// @}

    double translation[3];
public:
    MechanicalObject();

    virtual ~MechanicalObject();

    VecCoord* getX()  { return x;  }
    VecDeriv* getV()  { return v;  }
    VecDeriv* getF()  { return f;  }
    VecDeriv* getDx() { return dx; }

    const VecCoord* getX()  const { return x;  }
    const VecDeriv* getV()  const { return v;  }
    const VecDeriv* getF()  const { return f;  }
    const VecDeriv* getDx() const { return dx; }

    virtual BasicMechanicalModel* getMechanicalModel() { return this; }

    //virtual void addMapping(Core::BasicMapping *mMap);

    virtual void setMapping(Core::BasicMechanicalMapping* map);

    virtual void addMechanicalModel(Core::BasicMechanicalObject* mm);

    virtual void removeMechanicalModel(Core::BasicMechanicalObject* mm);

    virtual void addForceField(Core::ForceField* mFField);

    virtual void removeForceField(Core::ForceField* mFField);

    virtual void addConstraint(Core::Constraint* mConstraint);

    virtual void removeConstraint(Core::Constraint* mConstraint);

    virtual void resize(int vsize);

    virtual void init();

    virtual void beginIteration(double dt);

    virtual void endIteration(double dt);

    virtual void propagateX();

    virtual void propagateV();

    virtual void propagateDx();

    virtual void resetForce();

    virtual void accumulateForce();

    virtual void accumulateDf();

    virtual void applyConstraints();

    virtual void addMDx(); ///< f += M dx

    virtual void accFromF(); ///< dx = M^-1 f

    /// Set the behavior object currently owning this model
    virtual void setObject(Abstract::BehaviorModel* obj);

    virtual void setTopology(Topology* topo);

    virtual Topology* getTopology();

    virtual void setMass(Mass* m);

    virtual Mass* getMass();

    /// @name Integration related methods
    /// @{

    VecCoord* getVecCoord(unsigned int index);

    VecDeriv* getVecDeriv(unsigned int index);

    virtual void vAlloc(VecId v);

    virtual void vFree(VecId v);

    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0);

    virtual double vDot(VecId a, VecId b);

    virtual void setX(VecId v);

    virtual void setV(VecId v);

    virtual void setF(VecId v);

    virtual void setDx(VecId v);

    /// @}
};

} // namespace Core

} // namespace Sofa

#endif
