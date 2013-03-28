/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

/* Francois Faure, 2013 */

#include "Sofa_test.h"
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/common/VectorOperations.h>

namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace core;


/** Base class for the Mapping tests.
  The derived classes just need to create a mapping with an input and an output, set input positions and compute the expected output positions.
  This class provides the following:
  - test_apply() compare the actual output positions with the expected ones
  - test_applyJ() using small changes of the input
  */

template <typename _InDataTypes, typename _OutDataTypes>
struct Mapping_test : public Sofa_test<typename _InDataTypes::Real>
{
    typedef _InDataTypes In;
    typedef core::State<In> InState;
    typedef typename InState::Real  Real;
    typedef typename InState::VecCoord  InVecCoord;
    typedef typename InState::VecDeriv  InVecDeriv;
    typedef typename InState::ReadVecCoord  ReadInVecCoord;
    typedef typename InState::WriteVecCoord WriteInVecCoord;
    typedef typename InState::ReadVecDeriv  ReadInVecDeriv;
    typedef typename InState::WriteVecDeriv WriteInVecDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;

    typedef _OutDataTypes Out;
    typedef core::State<Out> OutState;
    typedef typename OutState::Coord     OutCoord;
    typedef typename OutState::VecCoord  OutVecCoord;
    typedef typename OutState::VecDeriv  OutVecDeriv;
    typedef typename OutState::ReadVecCoord  ReadOutVecCoord;
    typedef typename OutState::WriteVecCoord WriteOutVecCoord;
    typedef typename OutState::ReadVecDeriv  ReadOutVecDeriv;
    typedef typename OutState::WriteVecDeriv WriteOutVecDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;

    OutVecCoord expectedChildCoords;   ///< expected child positions after apply

    /** This tests assumes that init() has been applied,
      and that the expected output positions have been written in variable expectedChildCoords.
      It compares the actual output positions to the expected output positions.
    */
    virtual bool test_apply()
    {
        // apply has been done in the init();
        ReadOutVecCoord xout = toModel->readPositions();
//        ReadOutVecDeriv vout = outDofs->readVelocities();

        bool succeed=true;
        for( unsigned i=0; i<xout.size(); i++ )
        {
            OutCoord xdiff = xout[i] - expectedChildCoords[i];
            if( !isSmall(  xdiff.norm() ) ) {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: " << xout[i] <<", expected: " << expectedChildCoords[i];
                succeed = false;
            }
//            OutDeriv vdiff = vout[i] - expectedChildVels[i];
//            if( !isSmall(  vdiff.norm() ) ) {
//                ADD_FAILURE() << "Velocity of mapped particle " << i << " is wrong: " << vout[i] <<", expected: " << expectedChildVels[i];
//                succeed = false;
//            }
        }

        return succeed;
    }


    /** Test applyJ by comparing its result with a small displacement.
        This tests generate a small random perturbation dp of the parent positions,
        and checks that the change of child positions dc is accurately computed using the Jacobian: dc = J.dp

        If the sensitivity of the mapping is high (i.e. a small change of input generates a large change of output),
        and the mapping is nonlinear,
        then this test may return false even if applyJ works correctly.
        To avoid this, choose examples where the output changes have the same order of magnitude as the input changes.

    \param perturbation The maximum magnitude of the perturbation of each scalar value is perturbation * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    \param maxError The test is successfull if the difference (Linf norm of dc - J.dp) is less than  maxError * numeric_limits<Real>::epsilon
*/
    virtual bool test_applyJ( Real perturbation=1000, Real maxError=10 )
    {
        const MechanicalParams* mparams = MechanicalParams::defaultInstance();
        bool result = true;
        const unsigned Nin=fromModel->getSize(), Nout=toModel->getSize();

        // save current child positions
        OutVecCoord currentXout;
        {
            ReadOutVecCoord readCurrentXout = toModel->readPositions();
            currentXout.resize(Nout);
            for( unsigned i=0; i<Nout; i++ )
                currentXout[i] = readCurrentXout[i];
            // the ReadOutVecCoord will be destroyed
        }
//        cerr<<"currentXout = " << currentXout << endl;

        // increment parent positions
        WriteInVecCoord xIn = fromModel->writePositions();
        WriteInVecDeriv dxIn = fromModel->writeVelocities();
        InVecDeriv dxIn2(Nin);
        for( unsigned i=0; i<Nin; i++ )
        {
            dxIn[i] = dxIn2[i] = In::randomDeriv( this->epsilon() * perturbation );
            xIn[i] += dxIn[i];
        }

        // update child positions
        mapping->apply( mparams, core::VecCoordId::position(), core::VecCoordId::position() );

        // compute the difference
        OutVecCoord dxOut(Nout);
        ReadOutVecCoord readCurrentXout = toModel->readPositions();
//        cerr<<"new Xout = " << readCurrentXout << endl;
        for(unsigned i=0; i<Nout; i++ )
            dxOut[i] = readCurrentXout[i] - currentXout[i];
//        cerr<<"dxOut = " << dxOut << endl;

        // compute the difference using a linear approximation
        mapping->applyJ( mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );

        // compare
        ReadOutVecDeriv vOut= toModel->readVelocities();
        Real maxdiff = maxDiff(dxOut,vOut);
        if( maxdiff>this->epsilon()*maxError ){
            result = false;
            ADD_FAILURE() << "applyJ test failed";
        }

//        // test getJs()
//        const vector<sofa::defaulttype::BaseMatrix*>* jacobians = mapping->getJs();
//        if( jacobians->size() != 1 ){
////            FAIL()<< "Mapping->getJs() should have size == 1";
//            return false;
//        }
//        OutVecDeriv Jv(Nout);
//        (*jacobians)[0]->opMulV( &Jv, &dxIn2);
//        Real maxdiffJv = maxDiff(Jv,vOut);
//        if( maxdiffJv>this->epsilon()*maxError ){
//            result = false;
//            ADD_FAILURE() << "getJs() test failed";
//        }


        return result;
    }


protected:
    /// To be done by the derived class after the mapping is created and connected to its input and output.
    void setMapping( typename core::Mapping<In,Out>::SPtr m )
    {
        mapping = m.get();
        fromModel = mapping->getFromModel();
        toModel =   mapping->getToModel();
        if(!fromModel){
            ADD_FAILURE() << "Could not find fromModel";
        }
        if(!toModel){
            ADD_FAILURE() << "Could not find toModel";
        }
    }

    template<class C1, class C2>
    Real maxDiff( const C1& c1, const C2& c2 )
    {
        if( c1.size()!=c2.size() ){
            ADD_FAILURE() << "containers have different sizes";
            return this->infinity();
        }

        Real maxdiff = 0;
        for(unsigned i=0; i<c1.size(); i++ ){
//            cerr<< c2[i]-c1[i] << " ";
            if( (c1[i]-c2[i]).norm()>maxdiff )
                maxdiff = (c1[i]-c2[i]).norm();
        }
        return maxdiff;
    }

private:
    core::Mapping<In,Out>* mapping;
    InState* fromModel;
    OutState*  toModel;

};



} // namespace sofa
