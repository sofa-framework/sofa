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

    BasicMechanicalMapping* mapping;
    std::vector< Core::ForceField *> forcefields;
    std::vector< Core::BasicMechanicalObject *> mmodels;
    Topology* topology;
    Mass* mass;
    typedef typename std::vector< Core::ForceField* >::iterator ForceFieldIt;
    typedef typename std::vector< Core::BasicMechanicalObject* >::iterator MModelIt;

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
};

} // namespace Core

} // namespace Sofa

#endif
