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
#include <sofa/component/linearsolver/FullVector.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>

namespace sofa {

using std::cout;
using std::cout;
using std::endl;
using namespace core;


/** Base class for the Mapping tests.
  The derived classes just need to create a mapping with an input and an output, set input positions and compute the expected output positions.
  Then test_apply() compares the actual output positions with the expected ones.
  The jacobians are tested automatically in test_Jacobian() :
  - A small change of the input positions dxIn is randomly chosen and added to the current position. The same is set as velocity.
  - mapping->apply is called, and the difference dXout between the new output positions and the previous positions is computed
  - to validate mapping->applyJ, dXin is converted to input velocity vIn and mapping->applyJ is called. dXout and the output velocity vOut must be the same (up to linear approximations errors, thus we apply a very small change of position).
  - to validate mapping->getJs, we use it to get the Jacobian, then we check that J.vIn = vOut
  - to validate mapping->applyJT, we apply it after setting the child force fc=vOut, then we check that parent force fp = J^T.fc

  */

template <typename _InDataTypes, typename _OutDataTypes>
struct Mapping_test : public Sofa_test<typename _InDataTypes::Real>
{
    typedef _InDataTypes In;
    typedef core::State<In> InState;
    typedef typename InState::Real  Real;
    typedef typename InState::Deriv  InDeriv;
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

    typedef component::linearsolver::EigenSparseMatrix<In,Out> EigenSparseMatrix;

    OutVecCoord expectedChildCoords;   ///< expected child positions after apply

    /** This tests assumes that init() has been applied,
      and that the expected output positions have been written in variable expectedChildCoords.
      It compares the actual output positions to the expected output positions.
      The velocities are not tested, they will be in test_Jacobian()
    */
    virtual bool test_apply()
    {
        // apply has been done in the init();
        ReadOutVecCoord xout = toModel->readPositions();

        bool succeed=true;
        for( unsigned i=0; i<xout.size(); i++ )
        {
            OutCoord xdiff = xout[i] - expectedChildCoords[i];
            if( !isSmall(  xdiff.norm() ) ) {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: " << xout[i] <<", expected: " << expectedChildCoords[i];
                succeed = false;
            }
        }

        return succeed;
    }




    /** Test all the uses of the Jacobian of the mapping.
        applyJ is tested by comparing its result with a the consequence of a small displacement.

        The Jacobian matrix J is obtained using getJs(), and validated by comparison of its product with the result of applyJ().

        Matrix J is then used to validate applyJT.

        Function applyDJT() is validated by comparing the parent forces created by the same child forces, before and after the small displacement.

        The small displacement and the child forces are generated using random numbers.

        If the sensitivity of the mapping is high (i.e. a small change of input generates a large change of output),
        and the mapping is nonlinear,
        then this test may return false even if applyJ works correctly.
        To avoid this, choose examples where the output changes have the same order of magnitude as the input changes.

    \param perturbation The maximum magnitude of the change of each scalar value of the small displacement is perturbation * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    \param maxError The test is successfull if the (infinite norm of the) difference is less than  maxError * numeric_limits<Real>::epsilon
*/
    virtual bool test_Jacobian( Real perturbation=1000, Real maxError=10 )
    {
        MechanicalParams mparams;
        mparams.setKFactor(1.0);
        mparams.setSymmetricMatrix(false);
        bool result = true;
        const unsigned Np=fromModel->getSize(), Nc=toModel->getSize();

        InVecCoord xp(Np),xp1(Np);
        InVecDeriv vp(Np),fp(Np),dfp(Np),fp2(Np);
        OutVecCoord xc(Nc),xc1(Nc);
        OutVecDeriv vc(Nc),fc(Nc);

        // get position data
        copyFromData( xp,fromModel->readPositions() );
        copyFromData( xc,  toModel->readPositions() ); // positions and have already been propagated
//        cout<<"parent positions xp = "<< xp << endl;
//        cout<<"child  positions xc = "<< xc << endl;

        // set random child forces and propagate them to the parent
        for( unsigned i=0; i<Nc; i++ ){
            fc[i] = Out::randomDeriv( 1.0 );
        }
//        cout<<"random child forces  fc = "<<fc<<endl;
        copyToData( toModel->writeForces(), fc );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( fp, fromModel->readForces() );
//        cout<<"parent forces fp = "<<fp<<endl;

        // set small parent velocities and use them to update the child
        for( unsigned i=0; i<Np; i++ ){
            vp[i] = In::randomDeriv( this->epsilon() * perturbation );
        }
//        cout<<"parent velocities vp = " << vp << endl;
        for( unsigned i=0; i<Np; i++ ){             // and small displacements
            xp1[i] = xp[i] + vp[i];
        }
//        cout<<"new parent positions xp1 = " << xp1 << endl;

        // propagate small velocity
        copyToData( fromModel->writeVelocities(), vp );
        mapping->applyJ( &mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );
        copyFromData( vc, toModel->readVelocities() );
//        cout<<"child velocity vc = " << vc << endl;


        // apply geometric stiffness
        copyToData( fromModel->writeDx(), vp );
        dfp.fill( InDeriv() );
        copyToData( fromModel->writeForces(), dfp );
        mapping->applyDJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( dfp, fromModel->readForces() ); // fp + df due to geometric stiffness
//        cout<<"dfp = " << dfp << endl;

        // Jacobian will be obsolete after applying new positions
        EigenSparseMatrix* J = getMatrix(mapping->getJs());
//        cout<<"J = "<< endl << *J << endl;
        OutVecDeriv Jv(Nc);
        J->mult(Jv,vp);

        // ================ test applyJT()
        InVecDeriv jfc( (long)Np,InDeriv());
        J->addMultTranspose(jfc,fc);
//        cout<<"jfc = " << jfc << endl;
//        cout<<" fp = " << fp << endl;
        if( this->maxDiff(jfc,fp)>this->epsilon()*maxError ){
            result = false;
            ADD_FAILURE() << "applyJT test failed";
        }
        // ================ test getJs()
        // check that J.vp = vc
//        cout<<"vp = " << vp << endl;
//        cout<<"Jvp = " << Jv << endl;
//        cout<<"vc  = " << vc << endl;
        if( this->maxDiff(Jv,vc)>this->epsilon()*maxError ){
            result = false;
            ADD_FAILURE() << "getJs() test failed";
        }


        // propagate small displacement
        copyToData( fromModel->writePositions(), xp1 );
//        cout<<"new parent positions xp1 = " << xp1 << endl;
        mapping->apply ( &mparams, core::VecCoordId::position(), core::VecCoordId::position() );
        copyFromData( xc1, toModel->readPositions() );
//        cout<<"new child positions xc1 = " << xc1 << endl;

        // ================ test applyJ: compute the difference between propagated displacements and velocities
        OutVecCoord dxc(Nc),dxcv(Nc);
        for(unsigned i=0; i<Nc; i++ ){
            dxc[i] = xc1[i] - xc[i];
            dxcv[i] = vc[i]; // convert VecDeriv to VecCoord for comparison. Because strangely enough, Coord-Coord substraction returns a Coord (should be a Deriv)
        }
//        cout<<"dxc = " << dxc << endl;
        if( this->maxDiff(dxc,dxcv)>this->epsilon()*maxError ){
            result = false;
            ADD_FAILURE() << "applyJ test failed";
        }


        // update parent force based on the same child forces
        fp2.fill( InDeriv() );
        copyToData( fromModel->writeForces(), fp2 );  // reset parent forces before accumulating child forces
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( fp2, fromModel->readForces() );
//        cout<<"updated parent forces fp2 = "<< fp2 << endl;
        InVecDeriv fp12(Np);
        for(unsigned i=0; i<Np; i++){
            fp12[i] = fp2[i]-fp[i];       // fp2 - fp
        }
//        cout<<"fp2 - fp = " << fp12 << endl;



        // ================ test applyDJT()
        if( this->maxDiff(dfp,fp12)>this->epsilon()*maxError ){
            result = false;
            ADD_FAILURE() << "applyDJT test failed" << endl <<
                             "dfp    = " << dfp << endl <<
                             "fp2-fp = " << fp12 << endl;
        }


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
//            cout<< c2[i]-c1[i] << " ";
            if( (c1[i]-c2[i]).norm()>maxdiff )
                maxdiff = (c1[i]-c2[i]).norm();
        }
        return maxdiff;
    }

    /// Resize the Vector and copy it from the Data
    template<class Vector, class ReadData>
    void copyFromData( Vector& v, const ReadData& d){
        v.resize(d.size());
        for( unsigned i=0; i<v.size(); i++)
            v[i] = d[i];
    }

    /// Copy the Data from the Vector. They must have the same size.
    template<class WriteData, class Vector>
    void copyToData( WriteData d, const Vector& v){
        for( unsigned i=0; i<d.size(); i++)
            d[i] = v[i];
    }

    /// Get one EigenSparseMatrix out of a list. Error if not one single matrix in the list.
    static EigenSparseMatrix* getMatrix(const vector<sofa::defaulttype::BaseMatrix*>* matrices)
    {
        if( matrices->size() != 1 ){
            ADD_FAILURE()<< "Matrix list should have size == 1 in simple mappings";
//            return 0;
        }
        EigenSparseMatrix* ei = dynamic_cast<EigenSparseMatrix*>((*matrices)[0] );
        if( ei == NULL ){
            ADD_FAILURE() << "getJs returns a matrix of non-EigenSparseMatrix type";
        }
        return ei;
    }

private:
    core::Mapping<In,Out>* mapping;
    InState* fromModel;
    OutState*  toModel;

};



} // namespace sofa
