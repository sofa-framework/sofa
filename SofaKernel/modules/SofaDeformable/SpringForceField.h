/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPRINGFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/accessor.h>

#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/// This class contains the description of one linear spring
template<class T>
class LinearSpring
{
public:
    typedef T Real;
    int  m1, m2;            ///< the two extremities of the spring: masses m1 and m2
    Real ks;                ///< spring stiffness
    Real kd;                ///< damping factor
    Real initpos;           ///< rest length of the spring
    bool elongationOnly;    ///< only forbid elongation, not compression
    bool enabled;           ///< false to disable this spring (i.e. broken)

    LinearSpring(int m1=0, int m2=0, double ks=0.0, double kd=0.0, double initpos=0.0, bool noCompression=false, bool enabled=true)
        : m1(m1), m2(m2), ks((Real)ks), kd((Real)kd), initpos((Real)initpos), elongationOnly(noCompression), enabled(enabled)
    {
    }

    LinearSpring(int m1, int m2, float ks, float kd=0, float initpos=0, bool noCompression=false, bool enabled=true)
        : m1(m1), m2(m2), ks((Real)ks), kd((Real)kd), initpos((Real)initpos), elongationOnly(noCompression), enabled(enabled)
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
class SpringForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SpringForceField,DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField,DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef helper::ReadAccessor< Data< VecCoord > > RDataRefVecCoord;
    typedef helper::WriteAccessor< Data< VecCoord > > WDataRefVecCoord;
    typedef helper::ReadAccessor< Data< VecDeriv > > RDataRefVecDeriv;
    typedef helper::WriteAccessor< Data< VecDeriv > > WDataRefVecDeriv;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef LinearSpring<Real> Spring;

    Data<SReal> ks; ///< uniform stiffness for the all springs
    Data<SReal> kd; ///< uniform damping for the all springs
    Data<float> showArrowSize; ///< size of the axis
    Data<int> drawMode;             ///Draw Mode: 0=Line - 1=Cylinder - 2=Arrow
    Data<sofa::helper::vector<Spring> > springs; ///< pairs of indices, stiffness, damping, rest length

protected:
    core::objectmodel::DataFileName fileSprings;

protected:
    bool maskInUse;
    Real m_potentialEnergy;
    class Loader;

    SpringForceFieldInternalData<DataTypes> data;
    friend class SpringForceFieldInternalData<DataTypes>;

    virtual void addSpringForce(Real& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int /*i*/, const Spring& spring);


    SpringForceField(MechanicalState* object1, MechanicalState* object2, SReal _ks=100.0, SReal _kd=5.0);
    SpringForceField(SReal _ks=100.0, SReal _kd=5.0);

public:
    bool load(const char *filename);

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    const sofa::helper::vector< Spring >& getSprings() const {return springs.getValue();}

    virtual void reinit() override;
    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f1, DataVecDeriv& f2, const DataVecCoord& x1, const DataVecCoord& x2, const DataVecDeriv& v1, const DataVecDeriv& v2) override;
    virtual void addDForce(const core::MechanicalParams*, DataVecDeriv& df1, DataVecDeriv& df2, const DataVecDeriv& dx1, const DataVecDeriv& dx2 ) override;

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using Inherit::getPotentialEnergy;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& data_x1, const DataVecCoord& data_x2) const override;

    using Inherit::addKToMatrix;
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*kFact*/, unsigned int &/*offset*/);

    SReal getStiffness() const { return ks.getValue(); }
    SReal getDamping() const { return kd.getValue(); }
    void setStiffness(SReal _ks) { ks.setValue(_ks); }
    void setDamping(SReal _kd) { kd.setValue(_kd); }
    SReal getArrowSize() const {return showArrowSize.getValue();}
    void setArrowSize(float s) {showArrowSize.setValue(s);}
    int getDrawMode() const {return drawMode.getValue();}
    void setDrawMode(int m) {drawMode.setValue(m);}

    virtual void draw(const core::visual::VisualParams* vparams) override;

    // -- Modifiers

    void clear(int reserve=0)
    {
        sofa::helper::vector<Spring>& springs = *this->springs.beginEdit();
        springs.clear();
        if (reserve) springs.reserve(reserve);
        this->springs.endEdit();
    }

    void removeSpring(unsigned int idSpring)
    {
        if (idSpring >= (this->springs.getValue()).size())
            return;

        sofa::helper::vector<Spring>& springs = *this->springs.beginEdit();
        springs.erase(springs.begin() +idSpring );
        this->springs.endEdit();
    }

    void addSpring(int m1, int m2, SReal ks, SReal kd, SReal initlen)
    {
        springs.beginEdit()->push_back(Spring(m1,m2,ks,kd,initlen));
        springs.endEdit();
    }

    void addSpring(const Spring & spring)
    {
        springs.beginEdit()->push_back(spring);
        springs.endEdit();
    }

    virtual void updateForceMask() override;

    virtual void handleTopologyChange(core::topology::Topology *topo) override;

    /// initialization to export kinetic, potential energy  and force intensity to gnuplot files format
    virtual void initGnuplot(const std::string path) override;

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(SReal time) override;

    protected:
    /// stream to export Potential Energy to gnuplot files
    std::ofstream* m_gnuplotFileEnergy;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_DEFORMABLE_API LinearSpring<double>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec2dTypes>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec1dTypes>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_DEFORMABLE_API LinearSpring<float>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec2fTypes>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec1fTypes>;
extern template class SOFA_DEFORMABLE_API SpringForceField<defaulttype::Vec6fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPRINGFORCEFIELD_H */
