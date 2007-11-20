/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#ifndef SOFA_COMPONENT_FORCEFIELD_JOINTSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_JOINTSPRINGFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/PairInteractionForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <vector>
#include <sofa/defaulttype/Mat.h>
using namespace sofa::defaulttype;


namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class JointSpringForceFieldInternalData
{
public:
};

/** JointSpringForceField simulates 6D springs between Rigid DOFS
  Use kst vector to specify the directionnal stiffnesses (on each local axe)
  Use ksr vector to specify the rotational stiffnesses (on each local axe)
*/
template<class DataTypes>
class JointSpringForceField : public core::componentmodel::behavior::PairInteractionForceField<DataTypes>, public core::VisualModel
{
public:
    typedef typename core::componentmodel::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=Coord::static_size };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef Vec<N,Real> Vec;


    class Spring
    {
    public:
        int  m1, m2;			/// the two extremities of the spring: masses m1 and m2
        Real kd;				/// damping factor
        Vec  initTrans;		/// rest length of the spring
        Quat initRot;			/// rest orientation of the spring
        Quat lawfulTorsion;	/// general (lawful) torsion of the springs (used to fix a bug with large rotations)
        Quat extraTorsion;	/// extra (illicit) torsion of the springs (used to fix a bug with large rotations)

        sofa::defaulttype::Vec<6,bool> freeMovements;	///defines the axis where the movements is free. (0,1,2)--> translation axis (3,4,5)-->rotation axis
        Real softStiffnessTrans;	///stiffness to apply on axis where the translations are free (default 0.0)
        Real hardStiffnessTrans;	///stiffness to apply on axis where the translations are forbidden (default 10000.0)
        Real softStiffnessRot;	///stiffness to apply on axis where the rotations are free (default 0.0)
        Real hardStiffnessRot;	///stiffness to apply on axis where the rotations are forbidden (default 10000.0)

        Spring()
            : m1(0), m2(0), kd(0), lawfulTorsion(0,0,0,1), extraTorsion(0,0,0,1)
            , softStiffnessTrans(0), hardStiffnessTrans(10000), softStiffnessRot(0), hardStiffnessRot(10000)
        {
        }

        Spring(int m1, int m2, sofa::defaulttype::Vec<6,bool> freeAxis, Real softKst, Real hardKst, Real softKsr, Real hardKsr, Real kd)
            : m1(m1), m2(m2), kd(kd), lawfulTorsion(0,0,0,1), extraTorsion(0,0,0,1), freeMovements(freeAxis)
            , softStiffnessTrans(softKst), hardStiffnessTrans(hardKst), softStiffnessRot(softKsr), hardStiffnessRot(hardKsr)
        {
        }

        Spring(int m1, int m2, bool isFreeTx, bool isFreeTy, bool isFreeTz, bool isFreeRx, bool isFreeRy, bool isFreeRz, Real softKst, Real hardKst, Real softKsr, Real hardKsr, Real kd)
            : m1(m1), m2(m2), kd(kd), lawfulTorsion(0,0,0,1), extraTorsion(0,0,0,1), freeMovements(isFreeTx, isFreeTy, isFreeTz, isFreeRx, isFreeRy, isFreeRz)
            , softStiffnessTrans(softKst), hardStiffnessTrans(hardKst), softStiffnessRot(softKsr), hardStiffnessRot(hardKsr)
        {
        }

        Real getHardStiffnessRotation() {return hardStiffnessRot;}
        Real getSoftStiffnessRotation() {return softStiffnessRot;}

        void setHardStiffnessRotation(Real ksr) {	  hardStiffnessRot = ksr;  }
        void setSoftStiffnessRotation(Real ksr) {	  softStiffnessRot = ksr;  }

        void setFreeMovements(bool isFreeTx, bool isFreeTy, bool isFreeTz, bool isFreeRx, bool isFreeRy, bool isFreeRz)
        {
            freeMovements = sofa::defaulttype::Vec<6,bool>(isFreeTx, isFreeTy, isFreeTz, isFreeRx, isFreeRy, isFreeRz);
        }

        void setDamping(Real _kd) {  kd = _kd;	  }


        inline friend std::istream& operator >> ( std::istream& in, Spring& s )
        {
            in>>s.m1>>s.m2>>s.freeMovements>>s.softStiffnessTrans>>s.hardStiffnessTrans>>s.softStiffnessRot>>s.hardStiffnessRot>>s.kd>>s.initTrans>>s.initRot;
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const Spring& s )
        {
            out<<s.m1<<" "<<s.m2<<" "<<s.freeMovements<<" "<<s.softStiffnessTrans<<" "<<s.hardStiffnessTrans<<" "<<s.softStiffnessRot<<" "<<s.hardStiffnessRot<<" "<<s.kd<<" "<<s.initTrans<<" "<<s.initRot;
            return out;
        }

    };

protected:

    double m_potentialEnergy;
    /// general directional stiffness (use it to define the same stiffness on all springs)
    Data<Vec> kst;
    /// general rotational stiffness (use it to define the same stiffness on all springs)
    Data<Vec> ksr;
    /// general damping (use it to define the same damping on all springs)
    Data<double> kd;
    /// the list of the springs
    Data<sofa::helper::vector<Spring> > springs;
    /// the list of the local referentials of the springs
    VecCoord springRef;
    ///bool to allow the display of the 2 parts of springs torsions
    Data<bool> showLawfulTorsion;
    Data<bool> showExtraTorsion;


    JointSpringForceFieldInternalData<DataTypes> data;

    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce(double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, /*const*/ Spring& spring);
    /// Apply the stiffness, i.e. accumulate df given dx
    void addSpringDForce(VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, int i, const Spring& spring);

public:
    JointSpringForceField(MechanicalState* object1, MechanicalState* object2, Vec _kst=Vec(100.0,100.0,100.0), Vec _ksr=Vec(100.0,100.0,100.0), double _kd=5.0);
    JointSpringForceField(Vec _kst=Vec(100.0,100.0,100.0), Vec _ksr=Vec(100.0,100.0,100.0), double _kd=5.0);

    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::componentmodel::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    virtual void init();

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);

    virtual double getPotentialEnergy(const VecCoord&, const VecCoord&) { return m_potentialEnergy; }
    Vec getStiffnessTranslation() { return kst.getValue(); }
    Vec getStiffnessRotation() { return ksr.getValue(); }
    double getDamping() { return kd.getValue(); }
    void setStiffnessTranslation(Vec _kst) { kst.setValue(_kst); }
    void setStiffnessRotation(Vec _ksr) { ksr.setValue(_ksr); }
    void setDamping(double _kd) { kd.setValue(_kd); }

    sofa::helper::vector<Spring> * getSprings() { return springs.beginEdit(); }

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

    // -- Modifiers

    void clear(int reserve=0)
    {
        helper::vector<Spring>& springs = *this->springs.beginEdit();
        springs.clear();
        if (reserve) springs.reserve(reserve);
        this->springs.endEdit();
    }

    void addSpring(int m1, int m2, sofa::defaulttype::Vec<6,bool> freeAxis, Real softKs, Real hardKs, Real kd, Vec initLentghs, Vec initAngles)
    {
        Spring s(m1,m2,freeAxis, softKs, hardKs, softKs, hardKs, kd);
        s.initTrans = initLentghs;
        s.initRot = Quat::createFromRotationVector(initAngles);

        springs.beginEdit()->push_back(s);
        springs.endEdit();
    }

    void addSpring(int m1, int m2, sofa::defaulttype::Vec<6,bool> freeAxis, Real softKst, Real hardKst, Real softKsr, Real hardKsr, Real kd, Vec initLentghs, Vec initAngles)
    {
        Spring s(m1,m2,freeAxis, softKst, hardKst, softKsr, hardKsr, kd);
        s.initTrans = initLentghs;
        s.initRot = Quat::createFromRotationVector(initAngles);

        springs.beginEdit()->push_back(s);
        springs.endEdit();
    }
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
