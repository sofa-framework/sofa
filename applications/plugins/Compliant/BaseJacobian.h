/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef COMPLIANT_BaseJacobian_H
#define COMPLIANT_BaseJacobian_H

namespace sofa
{

namespace defaulttype
{


/** Template class used to implement one jacobian block
*/
template<class TIn, class TOut>
class BaseJacobianBlock
{
public:
    typedef TIn In;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Real Real;

    typedef TOut Out;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;

    // Called in Apply
    virtual void addapply( OutCoord& result, const InCoord& data )=0;
    // Called in ApplyJ
    virtual void addmult( OutDeriv& result,const InDeriv& data )=0;
    // Called in ApplyJT
    virtual void addMultTranspose( InDeriv& result, const OutDeriv& data )=0;
};


/** Template class used to implement a sparse jacobian matrix
currently: fixed number of columns
TO DO: change this to sparseEigenMatrix?
TO DO: implement masks
*/

template<class BaseJacobianBlock,int nbCol_>
class BaseJacobianMatrix
{
public:
    typedef typename BaseJacobianBlock::InCoord InCoord;
    typedef typename BaseJacobianBlock::InDeriv InDeriv;
    typedef typename BaseJacobianBlock::InVecCoord InVecCoord;
    typedef typename BaseJacobianBlock::InVecDeriv InVecDeriv;

    typedef typename BaseJacobianBlock::OutCoord OutCoord;
    typedef typename BaseJacobianBlock::OutDeriv OutDeriv;
    typedef typename BaseJacobianBlock::OutVecCoord OutVecCoord;
    typedef typename BaseJacobianBlock::OutVecDeriv OutVecDeriv;
    typedef typename BaseJacobianBlock::Real Real;

    enum { nbCol = nbCol_ };
    unsigned int nbRow;

    typedef vector<Vec<nbCol,BaseJacobianBlock> > Matrix;
    typedef vector<Vec<nbCol,unsigned int> > Ref;

    Matrix matrix;
    const Ref* ref;

    void resize(int nbRow_=0) { nbRow=nbRow_; matrix.resize(nbRow); }

    void init(const Ref* ref_)
    {
        ref=ref_;
        resize(ref->size());
    }

    void apply( OutVecCoord& result, const InVecCoord& data )
    {
        for(unsigned int i=0; i<nbRow; i++)
        {
            result[i]=OutCoord();
            for(unsigned int j=0; j<nbCol; j++)
            {
                unsigned int index=(*ref)[i][j];
                matrix[i][j].addapply(result[i],data[index]);
            }
        }
    }

    void mult( OutVecDeriv& result,const InVecDeriv& data )
    {
        for(unsigned int i=0; i<nbRow; i++)
        {
            result[i]=OutDeriv();
            for(unsigned int j=0; j<nbCol; j++)
            {
                unsigned int index=(*ref)[i][j];
                matrix[i][j].addmult(result[i],data[index]);
            }
        }
    }

    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data )
    {
        for(unsigned int i=0; i<nbRow; i++)
            for(unsigned int j=0; j<nbCol; j++)
            {
                unsigned int index=(*ref)[i][j];
                matrix[i][j].addMultTranspose(result[index],data[i]);
            }
    }

};



} // namespace defaulttype
} // namespace sofa



#endif
