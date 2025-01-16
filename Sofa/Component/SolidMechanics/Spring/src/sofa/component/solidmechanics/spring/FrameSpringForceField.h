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
#include <sofa/type/Vec.h>
#include <vector>
#include <sofa/type/Mat.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
class FrameSpringForceFieldInternalData
{
public:
};

/** FrameSpringForceField simulates 6D springs between moving frames
        Use stiffnessTrans vector to specify the directionnal stiffnesses (on each local axis)
        Use stiffnessRot vector to specify the rotational stiffnesses (on each local axis)
 */
template<class DataTypes>
class FrameSpringForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FrameSpringForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

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
    typedef type::Vec<N,Real> VecN;


    class Spring
    {
    public:
        int  m1, m2;             /// the two extremities of the spring: masses m1 and m2 ( indexes of the DOFs)
        VecN vec1, vec2;
        Real kd;                 /// damping factor

        Real stiffnessTrans;  ///stiffness to apply on axis where the translations are free (default 0.0)
        Real stiffnessRot;    ///stiffness to apply on axis where the rotations are free (default 0.0)

        ///constructors
        Spring ( int m1=0, int m2=0, Real softKst=0, Real softKsr=0, Real kd=0 )
            : m1 ( m1 ), m2 ( m2 ), kd ( kd ), stiffnessTrans ( softKst ), stiffnessRot ( softKsr )
        {
        }

        //accessors
        Real getStiffnessRotation() {return stiffnessRot;}
        Real getStiffnessTranslation() {return stiffnessTrans;}
        VecN getInitVec1() {return vec1;}

        //affectors
        void setStiffnessRotation ( Real ksr ) {    stiffnessRot = ksr;  }
        void setStiffnessTranslation ( Real kst ) { stiffnessTrans = kst;  }
        void setInitVec1 ( const VecN& l ) { vec1=l; }
        void setInitVec2 ( const VecN& l ) { vec2=l; }
        void setDamping ( Real _kd ) {  kd = _kd;   }


        inline friend std::istream& operator >> ( std::istream& in, Spring& s )
        {
            //default joint is a free rotation joint --> translation is bloqued, rotation is free

            std::string str;
            in>>str;
            if ( str == "BEGIN_SPRING" )
            {
                in>>s.m1>>s.m2; //read references
                in>>str;
                while ( str != "END_SPRING" )
                {
                    if ( str == "KS_T" )
                        in>>s.stiffnessTrans;
                    else if ( str == "KS_R" )
                        in>>s.stiffnessRot;
                    else if ( str == "KD" )
                        in>>s.kd;
                    else if ( str == "VEC1" )
                        in>>s.vec1;
                    else if ( str == "VEC2" )
                        in>>s.vec2;
                    else
                    {
                        msg_error("FrameSpringForceField")<<" Error parsing Spring : Unknown Attribute '"<<str << "'";
                        return in;
                    }

                    in>>str;
                }
            }

            return in;
        }

        friend std::ostream& operator << ( std::ostream& out, const Spring& s )
        {
            out<<"BEGIN_SPRING  "<<s.m1<<" "<<s.m2<<"  ";

            //if ( s.stiffnessTrans != 0.0 )
            out<<"KS_T "<<s.stiffnessTrans<<"  ";
            //if ( s.stiffnessRot != 0.0 )
            out<<"KS_R "<<s.stiffnessRot<<"  ";
            //if ( s.kd != 0.0 )
            out<<"KD "<<s.kd<<"  ";
            //if ( s.vec1!= VecN ( 0, 0, 0 ) )
            out<<"VEC1 "<<s.vec1<<"  ";
            //if ( s.vec2!= VecN ( 0, 0, 0 ) )
            out<<"VEC2 "<<s.vec2<<"  ";

            out<<"END_SPRING"<<std::endl;
            return out;
        }

    };
    // end inner class spring



protected:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector<Spring> > springs;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> showLawfulTorsion;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_SPRING()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> showExtraTorsion;

    SReal m_potentialEnergy;
    /// the list of the springs
    Data<sofa::type::vector<Spring> > d_springs;
    /// the list of the local referentials of the springs
    VecCoord springRef;
    /// bool to allow the display of the 2 parts of springs torsions
    Data<bool> d_showLawfulTorsion;
    Data<bool> d_showExtraTorsion; ///< display the illicit part of the joint rotation

    FrameSpringForceFieldInternalData<DataTypes> data;


    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce ( SReal& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring );
    /// Apply the stiffness, i.e. accumulate df given dx
    void addSpringDForce ( VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, int i, const Spring& spring );



    FrameSpringForceField();
    FrameSpringForceField(MechanicalState* object1, MechanicalState* object2);

public:
    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    void init() override;


    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&, const DataVecCoord& ) const override { return m_potentialEnergy; }

    sofa::type::vector<Spring> * getSprings() { return d_springs.beginEdit(); }

    void draw(const core::visual::VisualParams* vparams) override;

    // -- Modifiers

    void clear ( int reserve=0 );

    void addSpring ( const Spring& s );

    void addSpring ( int m1, int m2, Real softKst, Real softKsr, Real kd );

};

#if !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_FRAMESPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API FrameSpringForceField<defaulttype::Rigid3Types>;

#endif

} // namespace sofa::component::solidmechanics::spring
