/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_JOINTSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_JOINTSPRINGFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <vector>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{


template<typename DataTypes>
class JointSpring
{
public:

    typedef typename DataTypes::Coord    Coord   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Vec<N,Real> Vector;

    int  m1, m2;			/// the two extremities of the spring: masses m1 and m2
    Real kd;				/// damping factor
    Vector torsion;		/// torsion of the springs in axis/angle format
    Vector lawfulTorsion;	/// projected torsion in allowed angles
    Vector KT;	// linear stiffness
    Vector KR;	// angular stiffness
    defaulttype::Quat ref; // referential of the spring (p1) to use it in addSpringDForce()

    Vector  initTrans;		/// offset length of the spring
    defaulttype::Quat initRot;			/// offset orientation of the spring

    sofa::defaulttype::Vec<6,bool> freeMovements;	///defines the axis where the movements is free. (0,1,2)--> translation axis (3,4,5)-->rotation axis
    Real softStiffnessTrans;	///stiffness to apply on axis where the translations are free (default 0.0)
    Real hardStiffnessTrans;	///stiffness to apply on axis where the translations are forbidden (default 10000.0)
    Real softStiffnessRot;	///stiffness to apply on axis where the rotations are free (default 0.0)
    Real hardStiffnessRot;	///stiffness to apply on axis where the rotations are forbidden (default 10000.0)
    Real blocStiffnessRot;	///stiffness to apply on axis where the rotations are bloqued (=hardStiffnessRot/100)
    bool needToInitializeTrans;
    bool needToInitializeRot;

    sofa::defaulttype::Vec<6,Real> limitAngles; ///limit angles on rotation axis (default no limit)

    ///constructors
    JointSpring()
        : m1(0), m2(0), kd(0), torsion(0,0,0) ,  lawfulTorsion(0,0,0), KT(0,0,0) , KR(0,0,0)
        , softStiffnessTrans(0), hardStiffnessTrans(10000), softStiffnessRot(0), hardStiffnessRot(10000), blocStiffnessRot(100), needToInitializeTrans(true), needToInitializeRot(true)
        //, freeMovements(0,0,0,1,1,1), limitAngles(-100000, 100000, -100000, 100000, -100000, 100000)
    {
        freeMovements = sofa::defaulttype::Vec<6,bool>(0,0,0,1,1,1);
        limitAngles = sofa::defaulttype::Vec<6,Real>(-100000, 100000, -100000, 100000, -100000, 100000);
        initTrans = Vector(0,0,0);
        initRot = defaulttype::Quat(0,0,0,1);
    }

    JointSpring(int m1, int m2)
        : m1(m1), m2(m2), kd(0),  torsion(0,0,0), lawfulTorsion(0,0,0), KT(0,0,0) , KR(0,0,0)
        , softStiffnessTrans(0), hardStiffnessTrans(10000), softStiffnessRot(0), hardStiffnessRot(10000), blocStiffnessRot(100), needToInitializeTrans(true), needToInitializeRot(true)
        //, freeMovements(0,0,0,1,1,1), limitAngles(-100000, 100000, -100000, 100000, -100000, 100000)
    {
        freeMovements = sofa::defaulttype::Vec<6,bool>(0,0,0,1,1,1);
        limitAngles = sofa::defaulttype::Vec<6,Real>(-100000, 100000, -100000, 100000, -100000, 100000);
        initTrans = Vector(0,0,0);
        initRot = defaulttype::Quat(0,0,0,1);
    }

    JointSpring(int m1, int m2, Real softKst, Real hardKst, Real softKsr, Real hardKsr, Real blocKsr, Real axmin, Real axmax, Real aymin, Real aymax, Real azmin, Real azmax, Real kd)
        : m1(m1), m2(m2), kd(kd),  torsion(0,0,0), lawfulTorsion(0,0,0), KT(0,0,0) , KR(0,0,0)
        //,limitAngles(axmin,axmax,aymin,aymax,azmin,azmax)
        , softStiffnessTrans(softKst), hardStiffnessTrans(hardKst), softStiffnessRot(softKsr), hardStiffnessRot(hardKsr), blocStiffnessRot(blocKsr), needToInitializeTrans(true), needToInitializeRot(true)
    {
        limitAngles = sofa::defaulttype::Vec<6,Real>(axmin,axmax,aymin,aymax,azmin,azmax);
        freeMovements = sofa::defaulttype::Vec<6,bool>(false, false, false, true, true, true);
        for (unsigned int i=0; i<3; i++)
        {
            if(limitAngles[2*i]==limitAngles[2*i+1])
                freeMovements[3+i] = false;
        }
        initTrans = Vector(0,0,0);
        initRot = defaulttype::Quat(0,0,0,1);
    }

    //accessors
    Real getHardStiffnessRotation() {return hardStiffnessRot;}
    Real getSoftStiffnessRotation() {return softStiffnessRot;}
    Real getHardStiffnessTranslation() {return hardStiffnessTrans;}
    Real getSoftStiffnessTranslation() {return softStiffnessTrans;}
    Real getBlocStiffnessRotation() { return blocStiffnessRot; }
    sofa::defaulttype::Vec<6,Real> getLimitAngles() { return limitAngles;}
    sofa::defaulttype::Vec<6,bool> getFreeAxis() { return freeMovements;}
    Vector getInitLength() { return initTrans; }
    defaulttype::Quat getInitOrientation() { return initRot; }

    //affectors
    void setHardStiffnessRotation(Real ksr) {	  hardStiffnessRot = ksr;  }
    void setSoftStiffnessRotation(Real ksr) {	  softStiffnessRot = ksr;  }
    void setHardStiffnessTranslation(Real kst) { hardStiffnessTrans = kst;  }
    void setSoftStiffnessTranslation(Real kst) { softStiffnessTrans = kst;  }
    void setBlocStiffnessRotation(Real ksb) {	  blocStiffnessRot = ksb;  }
    void setLimitAngles(const sofa::defaulttype::Vec<6,Real>& lims)
    {
        limitAngles = lims;
        if(lims[0]==lims[1]) freeMovements[3]=false;
        if(lims[2]==lims[3]) freeMovements[4]=false;
        if(lims[4]==lims[5]) freeMovements[5]=false;
    }
    void setLimitAngles(Real minx, Real maxx, Real miny, Real maxy, Real minz, Real maxz)
    {
        limitAngles = sofa::defaulttype::Vec<6,Real>(minx, maxx, miny, maxy, minz, maxz);
        if(minx==maxx) freeMovements[3]=false;
        if(miny==maxy) freeMovements[4]=false;
        if(minz==maxz) freeMovements[5]=false;
    }
    void setInitLength( const Vector& l) { initTrans=l; }
    void setInitOrientation( const defaulttype::Quat& o) { initRot=o; }
    void setInitOrientation( const Vector& o) { initRot=defaulttype::Quat::createFromRotationVector(o); }
    void setFreeAxis(const sofa::defaulttype::Vec<6,bool>& axis) { freeMovements = axis; }
    void setFreeAxis(bool isFreeTx, bool isFreeTy, bool isFreeTz, bool isFreeRx, bool isFreeRy, bool isFreeRz)
    {
        freeMovements = sofa::defaulttype::Vec<6,bool>(isFreeTx, isFreeTy, isFreeTz, isFreeRx, isFreeRy, isFreeRz);
    }
    void setDamping(Real _kd) {  kd = _kd;	  }


    inline friend std::istream& operator >> ( std::istream& in, JointSpring<DataTypes>& s )
    {
        //default joint is a free rotation joint --> translation is bloqued, rotation is free
        s.freeMovements = sofa::defaulttype::Vec<6,bool>(false, false, false, true, true, true);
        s.initTrans = Vector(0,0,0);
        s.initRot = defaulttype::Quat(0,0,0,1);
        s.blocStiffnessRot = 0.0;
        //by default no angle limitation is set (bi values for initialisation)
        s.limitAngles = sofa::defaulttype::Vec<6,Real>(-100000., 100000., -100000., 100000., -100000., 100000.);
        bool initTransFound=false;
//		bool initRotFound=false;

        std::string str;
        in>>str;
        if(str == "BEGIN_SPRING")
        {
            in>>s.m1>>s.m2; //read references
            in>>str;
            while(str != "END_SPRING")
            {
                if(str == "FREE_AXIS")
                    in>>s.freeMovements;
                else if(str == "KS_T")
                    in>>s.softStiffnessTrans>>s.hardStiffnessTrans;
                else if(str == "KS_R")
                    in>>s.softStiffnessRot>>s.hardStiffnessRot;
                else if(str == "KS_B")
                    in>>s.blocStiffnessRot;
                else if(str == "KD")
                    in>>s.kd;
                else if(str == "R_LIM_X")
                    in>>s.limitAngles[0]>>s.limitAngles[1];
                else if(str == "R_LIM_Y")
                    in>>s.limitAngles[2]>>s.limitAngles[3];
                else if(str == "R_LIM_Z")
                    in>>s.limitAngles[4]>>s.limitAngles[5];
                else if(str == "REST_T")
                {
                    in>>s.initTrans;
                    initTransFound=true;
                }
                else if(str == "REST_R")
                {
                    in>>s.initRot;
//					  initRotFound=true;
                }
                else
                {
                    msg_error("JointSprintForceField")<<"Unknown Attribute while parsing '"<<str<<"'" ;
                    return in;
                }

                in>>str;
            }
        }

        s.needToInitializeTrans = initTransFound;
        s.needToInitializeRot = initTransFound;

        //if no blocStiffnessRot was specified (typically 0), we use hardStiffnessRot/100
        if(s.blocStiffnessRot == 0.0)
            s.blocStiffnessRot = s.hardStiffnessRot/100;

        //if limit angle were specified, free rotation axis are set from them
        for (unsigned int i=0; i<3; i++)
        {
            if(s.limitAngles[2*i]==s.limitAngles[2*i+1])
                s.freeMovements[3+i] = false;
        }

        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const JointSpring<DataTypes>& s )
    {
        out<<"BEGIN_SPRING  "<<s.m1<<" "<<s.m2<<"  ";

        if (s.freeMovements[0]!=false || s.freeMovements[1]!=false || s.freeMovements[2]!=false || s.freeMovements[3]!=true || s.freeMovements[4]!=true || s.freeMovements[5]!=true)
            out<<"FREE_AXIS "<<s.freeMovements<<"  ";
        if (s.softStiffnessTrans != 0.0 || s.hardStiffnessTrans != 10000.0)
            out<<"KS_T "<<s.softStiffnessTrans<<" "<<s.hardStiffnessTrans<<"  ";
        if (s.softStiffnessRot != 0.0 || s.hardStiffnessRot != 10000.0)
            out<<"KS_R "<<s.softStiffnessRot<<" "<<s.hardStiffnessRot<<"  ";
        if (s.blocStiffnessRot != s.hardStiffnessRot/100)
            out<<"KS_B "<<s.blocStiffnessRot<<"  ";
        if (s.kd != 0.0)
            out<<"KD "<<s.kd<<"  ";
        if (s.limitAngles[0]!=-100000 || s.limitAngles[1] != 100000)
            out<<"R_LIM_X "<<s.limitAngles[0]<<" "<<s.limitAngles[1]<<"  ";
        if (s.limitAngles[2]!=-100000 || s.limitAngles[3] != 100000)
            out<<"R_LIM_Y "<<s.limitAngles[2]<<" "<<s.limitAngles[3]<<"  ";
        if (s.limitAngles[4]!=-100000 || s.limitAngles[5] != 100000)
            out<<"R_LIM_Z "<<s.limitAngles[4]<<" "<<s.limitAngles[5]<<"  ";
        if (s.initTrans!= Vector(0,0,0))
            out<<"REST_T "<<s.initTrans<<"  ";
        if (s.initRot[3]!= 1)
            out<<"REST_R "<<s.initRot<<"  ";

        out<<"END_SPRING"<<std::endl;
        return out;
    }

};
// end class JointSpring


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
class JointSpringForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(JointSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;
    typedef defaulttype::Vec<N,Real> Vector;

    typedef JointSpring<DataTypes> Spring;

protected:

    SReal m_potentialEnergy;

    std::ifstream* infile;
    std::ofstream* outfile;


    JointSpringForceFieldInternalData<DataTypes> data;
    friend class JointSpringForceFieldInternalData<DataTypes>;


    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce(SReal& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, /*const*/ Spring& spring);
    /// Apply the stiffness, i.e. accumulate df given dx
    void addSpringDForce(VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, int i, /*const*/ Spring& spring, Real kFactor);

    // project torsion to Lawfulltorsion according to limitangles
    void projectTorsion(Spring& spring);



    JointSpringForceField(MechanicalState* object1, MechanicalState* object2);
    JointSpringForceField();

    virtual ~JointSpringForceField();

public:

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    virtual void init();
    virtual void bwdInit();

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 );

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2);

    virtual SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const { return m_potentialEnergy; }

    sofa::helper::vector<Spring> * getSprings() { return springs.beginEdit(); }

    void draw(const core::visual::VisualParams* vparams);
    void computeBBox(const core::ExecParams*  params, bool /*onlyVisible*/);

    // -- Modifiers

    void clear(int reserve=0)
    {
        helper::vector<Spring>& springs = *this->springs.beginEdit();
        springs.clear();
        if (reserve) springs.reserve(reserve);
        this->springs.endEdit();
    }

    void addSpring(const Spring& s)
    {
        springs.beginEdit()->push_back(s);
        springs.endEdit();
    }


    void addSpring(int m1, int m2, Real softKst, Real hardKst, Real softKsr, Real hardKsr, Real blocKsr, Real axmin, Real axmax, Real aymin, Real aymax, Real azmin, Real azmax, Real kd)
    {
        Spring s(m1,m2,softKst,hardKst,softKsr,hardKsr, blocKsr, axmin, axmax, aymin, aymax, azmin, azmax, kd);

        const VecCoord& x1= this->mstate1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord& x2= this->mstate2->read(core::ConstVecCoordId::position())->getValue();

        s.initTrans = x2[m2].getCenter() - x1[m1].getCenter();
        s.initRot = x2[m2].getOrientation()*x1[m1].getOrientation().inverse();

        springs.beginEdit()->push_back(s);
        springs.endEdit();
    }

    /// the list of the springs
    Data<sofa::helper::vector<Spring> > springs;
    sofa::core::objectmodel::DataFileName f_outfilename;
    sofa::core::objectmodel::DataFileName f_infilename;
    Data < Real > f_period;
    Data<bool> f_reinit;
    Real lastTime;

    /// bool to allow the display of the 2 parts of springs torsions
    Data<bool> showLawfulTorsion;
    Data<bool> showExtraTorsion;
    Data<Real> showFactorSize;

    virtual void updateForceMask();
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_JOINTSPRINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RIGID_API JointSpring<defaulttype::Rigid3dTypes>;
extern template class SOFA_RIGID_API JointSpringForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API JointSpring<defaulttype::Rigid3fTypes>;
extern template class SOFA_RIGID_API JointSpringForceField<defaulttype::Rigid3fTypes>;
#endif
#endif
} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_JOINTSPRINGFORCEFIELD_H */
