#ifndef SOFA_CORE_MECHANICALOBJECT_H
#define SOFA_CORE_MECHANICALOBJECT_H

#include "ForceField.h"
#include "BasicMapping.h"
#include "MechanicalModel.h"
#include <vector>
#include <assert.h>

namespace Sofa
{

namespace Core
{

template <class DataTypes>
class MechanicalObject : public MechanicalModel<DataTypes>
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

    std::vector< Core::ForceField *> forcefields;
    std::vector< Core::BasicMapping *> mappings;
    Topology* topology;
    typedef typename std::vector< Core::ForceField* >::iterator ForceFieldIt;
    typedef typename std::vector< Core::BasicMapping* >::iterator MappingIt;

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

    virtual void addMapping(Core::BasicMapping *mMap);

    virtual void addForceField(Core::ForceField *mFField);

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

    /// Set the behavior object currently ownning this model
    virtual void setObject(Abstract::BehaviorModel* obj);

    virtual void setTopology(Topology* topo);

    virtual Topology* getTopology();
};

} // namespace Core

} // namespace Sofa

#endif
