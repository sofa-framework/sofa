// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENTS_SPRINGFORCEFIELD_H
#define SOFA_COMPONENTS_SPRINGFORCEFIELD_H

#include "Sofa/Core/InteractionForceField.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Common/Vec.h"

#include <vector>

namespace Sofa
{

namespace Components
{

using namespace Common;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SpringForceFieldInternalData
{
public:
};

template<class DataTypes>
class SpringForceField : public Core::InteractionForceField, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:
    Core::MechanicalModel<DataTypes>* object1;
    Core::MechanicalModel<DataTypes>* object2;
    double m_potentialEnergy;
    DataField<double> ks;
    DataField<double> kd;

    class Spring
    {
    public:
        int     m1, m2;		// the two extremities of the spring: masses m1 and m2
        double  ks;			// spring stiffness
        double  kd;			// damping factor
        double  initpos;	// rest length of the spring

        Spring(int m1=0, int m2=0, double ks=0.0, double kd=0.0, double initpos=0.0)
            : m1(m1), m2(m2), ks(ks), kd(kd), initpos(initpos)
        {
        }
    };

    std::vector<Spring> springs;
    class Loader;

    SpringForceFieldInternalData<DataTypes> data;

    void addSpringForce(double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring);

public:
    SpringForceField(Core::MechanicalModel<DataTypes>* object1, Core::MechanicalModel<DataTypes>* object2, double _ks=100.0, double _kd=5.0)
        : object1(object1), object2(object2), ks(dataField(&ks,_ks,"stiffness","uniform stiffness for the all springs")), kd(dataField(&kd,_kd,"damping","uniform damping for the all springs"))
    {
    }

    SpringForceField(Core::MechanicalModel<DataTypes>* object, double _ks=100.0, double _kd=5.0)
        : object1(object), object2(object), ks(dataField(&ks,_ks,"stiffness","uniform stiffness for the all springs")), kd(dataField(&kd,_kd,"damping","uniform damping for the all springs"))
    {
    }

    bool load(const char *filename);

    Core::MechanicalModel<DataTypes>* getObject1() { return object1; }
    Core::MechanicalModel<DataTypes>* getObject2() { return object2; }
    Core::BasicMechanicalModel* getMechModel1() { return object1; }
    Core::BasicMechanicalModel* getMechModel2() { return object2; }

    virtual void init();
    void initFromTopology();

    virtual void addForce();

    virtual void addDForce();

    virtual double getPotentialEnergy() { return m_potentialEnergy; }
    double getStiffness() { return ks.getValue(); }
    double getDamping() { return kd.getValue(); }
    void setStiffness(double _ks) { ks.setValue(_ks); }
    void setDamping(double _kd) { kd.setValue(_kd); }

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

    // -- Modifiers

    void clear(int reserve=0)
    {
        springs.clear();
        if (reserve) springs.reserve(reserve);
    }

    void addSpring(int m1, int m2, double ks, double kd, double initlen)
    {
        springs.push_back(Spring(m1,m2,ks,kd,initlen));
    }
};

} // namespace Components

} // namespace Sofa

#endif
