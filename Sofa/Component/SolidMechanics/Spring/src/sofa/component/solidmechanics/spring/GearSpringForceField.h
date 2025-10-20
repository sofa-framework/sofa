/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <vector>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::component::solidmechanics::spring
{

template<typename DataTypes>
class GearSpring
{
public:

    typedef typename DataTypes::Coord    Coord   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    enum { N=DataTypes::spatial_dimensions };
    typedef type::Mat<N,N,Real> Mat;
    typedef sofa::type::Vec<N,Real> Vector;

    unsigned int  m1, m2;			/// the two extremities of the spring: masses m1 and m2
    unsigned int  p1, p2;			/// the two parents of each extremity
    Real previousAngle1, previousAngle2; // angle between parent and child in previous time step (modulo 2pi)
    Real angle1, angle2; // total angle between parent and child accumulated over time steps
    Coord ini1 , ini2; // use initial positions if parent = child

    Real kd;				/// damping factor

    sofa::type::Vec<2,unsigned int> freeAxis;	///defines the axis where the movements is free.
    Real hardStiffnessTrans;	///stiffness to apply on axis where the translations are forbidden (default 10000.0)
    Real softStiffnessRot;	///stiffness to apply on axis where the rotations are free (default 10000.0)
    Real hardStiffnessRot;	///stiffness to apply on axis where the rotations are forbidden (default 10000.0)
    Real Ratio;	/// Gear ratio (default 1)

    ///constructors
    GearSpring();
    GearSpring(unsigned int m1, unsigned int m2, unsigned int p1, unsigned int p2);
    GearSpring(unsigned int m1, unsigned int m2, unsigned int p1, unsigned int p2, Real hardKst, Real softKsr, Real hardKsr, Real kd, Real ratio);

    //accessors
    Real getHardStiffnessRotation() {return hardStiffnessRot;}
    Real getSoftStiffnessRotation() {return softStiffnessRot;}
    Real getHardStiffnessTranslation() {return hardStiffnessTrans;}
    sofa::type::Vec<2,unsigned int> getFreeAxis() { return freeAxis;}
    Real getRatio() {return Ratio;}

    //affectors
    void setHardStiffnessRotation(Real ksr) {	  hardStiffnessRot = ksr;  }
    void setSoftStiffnessRotation(Real ksr) {	  softStiffnessRot = ksr;  }
    void setHardStiffnessTranslation(Real kst) { hardStiffnessTrans = kst;  }
    void setRatio(Real ratio) { Ratio = ratio;  }

    void setFreeAxis(const sofa::type::Vec<2,unsigned int>& axis) { freeAxis = axis; }
    void setFreeAxis(unsigned int axis1, unsigned int axis2)
    {
        freeAxis = sofa::type::Vec<2,unsigned int>(axis1, axis2);
    }
    void setDamping(Real _kd) {  kd = _kd;	  }


    inline friend std::istream& operator >> ( std::istream& in, GearSpring<DataTypes>& s )
    {
        //default Gear is a Gear around x
        s.freeAxis = sofa::type::Vec<2,unsigned int>(0,0);

        std::string str;
        in>>str;
        if(str == "BEGIN_SPRING")
        {
            in>>s.p1>>s.m1>>s.p2>>s.m2; //read references
            in>>str;
            while(str != "END_SPRING")
            {
                if(str == "AXIS")
                    in>>s.freeAxis;
                else if(str == "KS_T")
                    in>>s.hardStiffnessTrans;
                else if(str == "KS_R")
                    in>>s.softStiffnessRot>>s.hardStiffnessRot;
                else if(str == "KD")
                    in>>s.kd;
                else if(str == "RATIO")
                    in>>s.Ratio;
                else
                {
                    msg_error("GearSpring")<<"parsing Spring : Unknown Attribute "<<str;
                    return in;
                }

                in>>str;
            }
        }

        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const GearSpring<DataTypes>& s )
    {
        out<<"BEGIN_SPRING  "<<s.p1<<" "<<s.m1<<" "<<s.p2<<" "<<s.m2<<"  ";

        if (s.freeAxis[0]!=0 || s.freeAxis[1]!=0 )
            out<<"AXIS "<<s.freeAxis<<"  ";
        if (s.hardStiffnessTrans != 10000.0)
            out<<"KS_T "<<s.hardStiffnessTrans<<"  ";
        if (s.softStiffnessRot != 10000.0 || s.hardStiffnessRot != 10000.0)
            out<<"KS_R "<<s.softStiffnessRot<<" "<<s.hardStiffnessRot<<"  ";
        if (s.kd != 0.0)
            out<<"KD "<<s.kd<<"  ";
        if (s.Ratio != 1.0)
            out<<"RATIO "<<s.Ratio<<"  ";

        out<<"END_SPRING"<<std::endl;
        return out;
    }

};
// end class GearSpring


template<class DataTypes>
class GearSpringForceFieldInternalData
{
public:
};

/** GearSpringForceField simulates 6D springs between Rigid DOFS
  Use kst vector to specify the directionnal stiffnesses (on each local axe)
  Use ksr vector to specify the rotational stiffnesses (on each local axe)
*/
template<class DataTypes>
class GearSpringForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(GearSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

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
    typedef type::Mat<N,N,Real> Mat;
    typedef sofa::type::Vec<N,Real> Vector;

    typedef GearSpring<DataTypes> Spring;

protected:

    SReal m_potentialEnergy;

    std::ofstream* outfile;

    GearSpringForceFieldInternalData<DataTypes> data;
    friend class GearSpringForceFieldInternalData<DataTypes>;

    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce(SReal& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, /*const*/ Spring& spring);
    /// Apply the stiffness, i.e. accumulate df given dx
    void addSpringDForce(VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, int i, /*const*/ Spring& spring, Real kFactor);

    GearSpringForceField();
    GearSpringForceField(MechanicalState* object1, MechanicalState* object2);

    virtual ~GearSpringForceField();

public:

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    void init() override;
    void reinit() override;
    void bwdInit() override;

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const override { return m_potentialEnergy; }

    sofa::type::vector<Spring> * getSprings() { return d_springs.beginEdit(); }

    void draw(const core::visual::VisualParams* vparams) override;

    // -- Modifiers

    void clear(int reserve=0)
    {
        type::vector<Spring>& springs = *this->d_springs.beginEdit();
        springs.clear();
        if (reserve) springs.reserve(reserve);
        this->d_springs.endEdit();
    }

    void addSpring(const Spring& s)
    {
        d_springs.beginEdit()->push_back(s);
        d_springs.endEdit();
    }


    void addSpring(int m1, int m2, int p1, int p2, Real hardKst, Real softKsr, Real hardKsr, Real kd, Real ratio)
    {
        Spring s(m1,m2,p1,p2,hardKst,softKsr,hardKsr, kd, ratio);

        d_springs.beginEdit()->push_back(s);
        d_springs.endEdit();
    }



    inline typename GearSpringForceField<DataTypes>::Real getAngleAroundAxis(Coord p1,Coord p2,unsigned int axis)
    {
        Mat M1, M2;
        p1.writeRotationMatrix(M1);
        p2.writeRotationMatrix(M2);
        Vector u,v;
        // compute angle as average angles between other axis
        Real angle = 0.0;
        unsigned int count = 0;
        Vector W,w;
        for(unsigned int i=0; i<u.size(); ++i) W[i]=M1(i,axis); // ref axis
        for(unsigned int j=0; j<u.size(); ++j) if(j!=axis)
            {
                for(unsigned int i=0; i<u.size(); ++i)  {u[i]=M1(i,j); v[i]=M2(i,j);}
                count++;  getVectorAngle(u,v,w); if(dot(w,W)<0) angle -= w.norm(); else angle += w.norm();
            }
        angle /= (Real)count;
        return angle;
    }

    inline void getVectorAngle(Vector u,Vector v,Vector &w)
    {
        w=cross(u,v);
        Real nw=w.norm(),dt=dot(u,v);
        if(nw) w*=atan2(nw,dt)/nw;
    }


    inline void getVectorAngle(Coord p1,Coord p2,unsigned int axis,Vector &w)
    {
        Mat M1, M2;
        p1.writeRotationMatrix(M1);
        p2.writeRotationMatrix(M2);
        Vector u,v;
        for(unsigned int i=0; i<u.size(); ++i) { u[i]=M1(i,axis); v[i]=M2(i,axis); }
        getVectorAngle(u,v,w);
    }

    /// the list of the springs
    Data<sofa::type::vector<Spring> > d_springs;
    sofa::core::objectmodel::DataFileName d_filename; ///< output file name
    Data < Real > d_period; ///< period between outputs
    Data<bool> d_reinit; ///< flag enabling reinitialization of the output file at each timestep
    Real lastTime;

    /// bool to allow the display of the extra torsion
    Data<Real> d_showFactorSize;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_GEARSPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API GearSpring<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API GearSpringForceField<defaulttype::Rigid3Types>;

#endif
} // namespace sofa::component::solidmechanics::spring
