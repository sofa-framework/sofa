/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/PairInteractionForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/component/component.h>


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

/// This class contains the description of one linear spring
template<class T>
class LinearSpring
{
public:
    typedef T Real;
    int     m1, m2;  ///< the two extremities of the spring: masses m1 and m2
    Real  ks;      ///< spring stiffness
    Real  kd;      ///< damping factor
    Real  initpos; ///< rest length of the spring

    LinearSpring(int m1=0, int m2=0, double ks=0.0, double kd=0.0, double initpos=0.0)
        : m1(m1), m2(m2), ks((Real)ks), kd((Real)kd), initpos((Real)initpos)
    {
    }

    LinearSpring(int m1, int m2, float ks, float kd=0, float initpos=0)
        : m1(m1), m2(m2), ks((Real)ks), kd((Real)kd), initpos((Real)initpos)
    {
    }

    inline friend std::istream& operator >> ( std::istream& in, LinearSpring<Real>& s )
    {
        in>>s.m1>>s.m2>>s.ks>>s.kd>>s.initpos;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const LinearSpring<Real>& s )
    {
        out<<s.m1<<" "<<s.m2<<" "<<s.ks<<" "<<s.kd<<" "<<s.initpos<<"\n";
        return out;
    }

};


/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SpringForceFieldInternalData
{
public:
};

/// Set of simple springs between particles
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

    typedef LinearSpring<Real> Spring;

protected:
    bool maskInUse;
    SReal m_potentialEnergy;
    Data<SReal> ks;
    Data<SReal> kd;
    Data<sofa::helper::vector<Spring> > springs;
    class Loader;

    SpringForceFieldInternalData<DataTypes> data;
    friend class SpringForceFieldInternalData<DataTypes>;

    void addSpringForce(SReal& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring);
    void updateMaskStatus();
public:
    SpringForceField(MechanicalState* object1, MechanicalState* object2, SReal _ks=100.0, SReal _kd=5.0);
    SpringForceField(SReal _ks=100.0, SReal _kd=5.0);

    virtual bool canPrefetch() const { return false; }

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    bool load(const char *filename);

    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    sofa::helper::vector< Spring > getSprings() {return springs.getValue();}

    virtual void reinit();
    virtual void init();

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);
    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);
    virtual double getPotentialEnergy(const VecCoord&, const VecCoord&) { return m_potentialEnergy; }


    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, double /*kFact*/, unsigned int &/*offset*/);

    SReal getStiffness() { return ks.getValue(); }
    SReal getDamping() { return kd.getValue(); }
    void setStiffness(SReal _ks) { ks.setValue(_ks); }
    void setDamping(SReal _kd) { kd.setValue(_kd); }

    void draw();

    // -- Modifiers

    void clear(int reserve=0)
    {
        sofa::helper::vector<Spring>& springs = *this->springs.beginEdit();
        springs.clear();
        if (reserve) springs.reserve(reserve);
        this->springs.endEdit();
    }

    void addSpring(int m1, int m2, SReal ks, SReal kd, SReal initlen)
    {
        springs.beginEdit()->push_back(Spring(m1,m2,ks,kd,initlen));
        springs.endEdit();
        updateMaskStatus();
    }

    virtual void handleTopologyChange(core::componentmodel::topology::Topology *topo);

    virtual bool useMask();


};

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec1dTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec1fTypes>;
extern template class SOFA_COMPONENT_FORCEFIELD_API SpringForceField<defaulttype::Vec6fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
