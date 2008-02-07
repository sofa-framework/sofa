// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/PairInteractionForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SpringForceFieldInternalData
{
public:
};

/** Define a set of springs between particles */
template<class DataTypes>
class SpringForceField : public core::componentmodel::behavior::PairInteractionForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename core::componentmodel::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

    class Spring
    {
    public:
        int     m1, m2;  ///< the two extremities of the spring: masses m1 and m2
        double  ks;      ///< spring stiffness
        double  kd;      ///< damping factor
        double  initpos; ///< rest length of the spring

        Spring(int m1=0, int m2=0, double ks=0.0, double kd=0.0, double initpos=0.0)
            : m1(m1), m2(m2), ks(ks), kd(kd), initpos(initpos)
        {
        }

        inline friend std::istream& operator >> ( std::istream& in, Spring& s )
        {
            in>>s.m1>>s.m2>>s.ks>>s.kd>>s.initpos;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Spring& s )
        {
            out<<s.m1<<" "<<s.m2<<" "<<s.ks<<" "<<s.kd<<" "<<s.initpos;
            return out;
        }

    };
protected:

    double m_potentialEnergy;
    Data<double> ks;
    Data<double> kd;
    Data<sofa::helper::vector<Spring> > springs;
    class Loader;

    SpringForceFieldInternalData<DataTypes> data;

    void addSpringForce(double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring);

public:
    SpringForceField(MechanicalState* object1, MechanicalState* object2, double _ks=100.0, double _kd=5.0);
    SpringForceField(double _ks=100.0, double _kd=5.0);

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    bool load(const char *filename);

    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    virtual void init();

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);
    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);
    virtual double getPotentialEnergy(const VecCoord&, const VecCoord&) { return m_potentialEnergy; }
    double getStiffness() { return ks.getValue(); }
    double getDamping() { return kd.getValue(); }
    void setStiffness(double _ks) { ks.setValue(_ks); }
    void setDamping(double _kd) { kd.setValue(_kd); }

    void draw();

    // -- Modifiers

    void clear(int reserve=0)
    {
        sofa::helper::vector<Spring>& springs = *this->springs.beginEdit();
        springs.clear();
        if (reserve) springs.reserve(reserve);
        this->springs.endEdit();
    }

    void addSpring(int m1, int m2, double ks, double kd, double initlen)
    {
        springs.beginEdit()->push_back(Spring(m1,m2,ks,kd,initlen));
        springs.endEdit();
    }
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
