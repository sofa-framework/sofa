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
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/EigenBaseSparseMatrix.h>
#include <sofa/type/Mat.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/helper/SortedPermutation.h>
#include <sofa/type/vector.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <sofa/helper/OwnershipSPtr.h>

namespace sofa::linearalgebra
{
using type::vector;


/** Variant of EigenBaseSparseMatrix, capable of block-view access.
  The blocks correspond to matrix blocks of the size of the DataTypes Deriv.

  There are two ways of filling the matrix:
  - Random block access is provided by method wBlock. Use compress() after the last insertion.
  - Block rows can be efficiently appended using methods beginBlockRow, createBlock, endBlockRow. Use compress() after the last insertion. The rows must be created in increasing index order.

  The two ways of filling the matrix can not be used at the same time.
  */
template<class InDataTypes, class OutDataTypes>
class EigenSparseMatrix : public EigenBaseSparseMatrix<typename OutDataTypes::Real>
{
public:
    typedef EigenBaseSparseMatrix<typename OutDataTypes::Real> Inherit;
    typedef typename OutDataTypes::Real OutReal;

    typedef typename OutDataTypes::Real Real; // what's inside the eigen matrix

    typedef Eigen::SparseMatrix<OutReal,Eigen::RowMajor> CompressedMatrix;
    typedef Eigen::Matrix<OutReal,Eigen::Dynamic,1>  VectorEigenOut;

    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Real InReal;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
    typedef type::Mat<Nout,Nin, OutReal> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.

    using Index = linearalgebra::BaseMatrix::Index;
protected:
    typedef std::map<Index,Block> BlockRowMap;        ///< Map which represents one block-view row of the matrix. The index represents the block-view column index of an entry.
    typedef std::map<Index,BlockRowMap> BlockMatMap;  ///< Map which represents a block-view matrix. The index represents the block-view index of a block-view row.
    BlockMatMap incomingBlocks;                     ///< To store block-view data before it is compressed in optimized format.
    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  VectorEigenIn;
    
    // some helpers to ease mapping from/to eigen types, lvalue and rvalue
    template<typename VecType>
    struct map_traits {
        typedef typename VecType::value_type value_type;
        static const unsigned size = value_type::total_size;
        typedef typename value_type::value_type real_type;
        typedef Eigen::Matrix< real_type, Eigen::Dynamic, 1 > matrix_type;

        typedef Eigen::Map< matrix_type > map_type;
        typedef Eigen::Map< const matrix_type > const_map_type;

        static map_type map(real_type* data, unsigned k) {
            return map_type(data, k * size);
        }

        static const_map_type const_map(const real_type* data, unsigned k) {
            return const_map_type(data, k * size);
        }

    };

    template<class VecDeriv>
    static typename map_traits<VecDeriv>::const_map_type map(const VecDeriv& data)
    {
        return map_traits<VecDeriv>::const_map(&data[0][0], data.size());
    }

    template<class VecDeriv>
    static typename map_traits<VecDeriv>::map_type map(VecDeriv& data)
    {
        return map_traits<VecDeriv>::map(&data[0][0], data.size());
    }

	template<class LHS, class RHS>
	static bool alias(const LHS& lhs, const RHS& rhs) {
		return (void*)(&lhs[0][0]) == (void*)(&rhs[0][0]);
	}

public:

    EigenSparseMatrix(int nbRow=0, int nbCol=0):Inherit(nbRow,nbCol) {}

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlockRows, int nbBlockCols)
    {
        this->resize(nbBlockRows * Nout, nbBlockCols * Nin);
    }

    /// Schedule the addition of the block at the given place. Scheduled additions must be finalized using function compress().
    void addBlock( unsigned row, unsigned col, const Block& b )
    {
        for( unsigned r=0; r<Nout; r++ )
            for( unsigned c=0; c<Nin; c++ )
                this->add( r + row*Nout, c + col*Nin, b(r,c) );
    }

    /// Insert ASAP in the compressed matrix. There must be no value at this place already.
    /// @warning basically works only if there is only one block on the row
    /// @warning empty rows should be created with a call to beginBlockRow + endSortedBlockRow
    void insertBackBlock( unsigned row, unsigned col, const Block& b )
    {
        for( unsigned r=0; r<Nout; r++ )
        {
            this->beginRow( r + row*Nout );
            for( unsigned c=0; c<Nin; c++ )
                this->insertBack( r + row*Nout, c + col*Nin, b(r,c) );
        }
    }


    /** Prepare the insertion of a new row of blocks in the matrix.
        Then create blocks using createBlock( unsigned column,  const Block& b ).
        Then finally use endBlockRow() or endSortedBlockRow() to validate the row insertion.
        @sa createBlock( unsigned column,  const Block& b )
        @sa endBlockRow()
        @warning empty rows should be created with a call to beginBlockRow + endSortedBlockRow
        */
    void beginBlockRow(unsigned row)
    {
        bRow = row;
        bColumns.clear();
        blocks.clear();
    }

    /** Create a block in the current row, which must be previously
        initialized using beginBlockRow(unsigned row).

        If the blocks are NOT created in column order, call endBlockRow().
        If the blocks are given in column order, endSortedBlockRow() will
        be more efficient.

        The blocks are not actually created in the matrix until method
        endBlockRow()/endSortedBlockRow() is called.

        @warning the block must NOT already exist
        */
    void createBlock( unsigned column,  const Block& b )
    {
        blocks.push_back(b);
        bColumns.push_back(column);
    }

    /** Finalize the creation of the current block row.
      @sa beginBlockRow(unsigned row)
      @sa createBlock( unsigned column,  const Block& b )

      If the block have been given in column order,
      endSortedBlockRow() is more efficient.
      */
    void endBlockRow()
    {
        vector<unsigned> p = helper::sortedPermutation(bColumns); // indices in ascending column order

        for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
        {
            this->beginRow(r+ bRow*Nout);
            for(unsigned i=0; i<p.size(); i++ )  // process the blocks in ascending order
            {
                const Block& b = blocks[p[i]];
                for( unsigned c=0; c<Nin; c++ )
                {
                    this->insertBack( r + bRow*Nout, c + bColumns[p[i]] * Nin, b(r,c));
                }
            }
        }
    }


    /** Finalize the creation of the current block row with
     * blocks given in column order.
      @sa beginBlockRow(unsigned row)
      @sa createBlock( unsigned column,  const Block& b ) in column order
      */
    void endSortedBlockRow()
    {
        for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
        {
            this->beginRow(r+ bRow*Nout);
            for(unsigned i=0; i<bColumns.size(); i++ )  // process the blocks in ascending order
            {
                const Block& b = blocks[i];
                for( unsigned c=0; c<Nin; c++ )
                {
                    this->insertBack( r + bRow*Nout, c + bColumns[i] * Nin, b(r,c));
                }
            }
        }
    }



    // max: added template real type to work around the mess between
    // mapping ::Real member types (sometimes it's In::Real, sometimes it's
    // Out::Real, see e.g. BarycentricMapping/RigidMapping)

    /** Set from a CompressedRowSparseMatrix. @pre crs must be compressed
      */
    template<class AnyReal>
    void copyFrom( const CompressedRowSparseMatrix< type::Mat<Nout,Nin, AnyReal> >& crs )
    {
        this->resize( crs.rowSize(), crs.colSize() );

//        int rowStarted = 0;
        for (unsigned int xi = 0; xi < crs.rowIndex.size(); ++xi)  // for each non-null block row
        {
            int blRow = crs.rowIndex[xi];      // block row

            typename CompressedRowSparseMatrix<Block>::Range rowRange(crs.rowBegin[xi], crs.rowBegin[xi+1]);

            for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
            {
                if(r+ blRow*Nout >= (unsigned)this->rowSize() ) break;
//                this->compressedMatrix.startVec(rowStarted++);


                for (Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
                {
                    int blCol = crs.colsIndex[xj];     // block column
                    const Block& b = crs.colsValue[xj]; // block value
                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < (unsigned)this->colSize() )
                        {
                        this->add(r + blRow*Nout, c + blCol*Nin, b(r,c));
//                        this->compressedMatrix.insertBack(r + blRow*Nout, c + blCol*Nin) = b(r,c);
                        }

                }
            }
        }
        this->compress();

    }

protected:

	// max: factored out the two exact same implementations for
	// VecDeriv/Data<VecDeriv>
	
	template<class OutType, class InType>
	void mult_impl(OutType& result, const InType& data) const {

        if( data.empty() ) return;
		 
		// use optimized product if possible
        if(canCast(data)) {

#if (SOFA_LINEARALGEBRA_HAVE_OPENMP == 1)
            if( alias(result, data) )
                map(result) = linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix, map(data).template cast<Real>() );
            else
                map(result).noalias() = linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix, map(data).template cast<Real>() );
#else
            if( alias(result, data) ) {
                this->map(result) = (this->compressedMatrix *
                                     this->map(data).template cast<Real>()).template cast<OutReal>();
            } else {
                this->map(result).noalias() = (this->compressedMatrix *
                                               this->map(data).template cast<Real>()).template cast<OutReal>();
            }
#endif
			
			return;
		}
		// convert the data to Eigen type
        VectorEigenOut aux1(this->colSize(),1), aux2(this->rowSize(),1);
        for(size_t i = 0, n = data.size(); i < n; ++i) {
			for(unsigned j = 0; j < Nin; ++j) {
                aux1[Nin * i + j] = data[i][j];
			}
		}
		
        // compute the product
#if (SOFA_LINEARALGEBRA_HAVE_OPENMP == 1)
        aux2.noalias() = linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix, aux1 );
#else
        aux2.noalias() = this->compressedMatrix * aux1;
#endif
        
        // convert the result back to the Sofa type
        for(size_t i = 0, n = result.size(); i < n; ++i) {
			for(unsigned j = 0; j < Nout; ++j) {
                result[i][j] = aux2[Nout * i + j];
			}
		}
	}


	template<class OutType, class InType>
	void addMult_impl( OutType& result, const InType& data, Real fact) const {
		
        if( data.empty() ) return;

		// use optimized product if possible
		if( canCast(data) ) {

#if (SOFA_LINEARALGEBRA_HAVE_OPENMP == 1)
            if( alias(result, data) )
                map(result) += linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix, this->map(data).template cast<Real>() * fact ).template cast<OutReal>();
            else
            {
                typename map_traits<OutType>::map_type r = map(result);
                r.noalias() += linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix, this->map(data).template cast<Real>() * fact ).template cast<OutReal>();
            }
#else
			// TODO multiply only the smallest dimension by fact 
            if( alias(result, data) ) {
                map(result) += (this->compressedMatrix * (map(data).template cast<Real>() * fact)).template cast<OutReal>();
            } else {
                auto r = map(result);
                r.noalias() += (this->compressedMatrix * (map(data).template cast<Real>() * fact)).template cast<OutReal>();
            }
#endif
			
			return;
		}

		// convert the data to Eigen type
        VectorEigenOut aux1(this->colSize()),aux2(this->rowSize());
        for(unsigned i = 0, n = data.size(); i < n; ++i) {
			for(unsigned j = 0; j < Nin; ++j) {
                aux1[Nin * i + j] = data[i][j];
			}
		}
        
        // compute the product
#if (SOFA_LINEARALGEBRA_HAVE_OPENMP == 1)
        aux2.noalias() = linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix, aux1 );
#else
        aux2.noalias() = this->compressedMatrix * aux1;
#endif
        
        // convert the result back to the Sofa type
        for(unsigned i = 0, n = result.size(); i < n; ++i) {
			for(unsigned j = 0; j < Nout; ++j) {
                result[i][j] += aux2[Nout * i + j] * fact;
			}
		}
	}

	template<class InType, class OutType>
    void addMultTranspose_impl( InType& result, const OutType& data, Real fact) const {

        if( data.empty() ) return;

		// use optimized product if possible
		if(canCast(result)) {

#if (SOFA_LINEARALGEBRA_HAVE_OPENMP == 1)
            if( alias(result, data) )
                map(result) += linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix.transpose(), this->map(data).template cast<Real>() * fact ).template cast<InReal>();
            else {
                typename map_traits<InType>::map_type r = map(result);
                r.noalias() += linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix.transpose(), this->map(data).template cast<Real>() * fact ).template cast<InReal>();
            }
#else
            // TODO multiply only the smallest dimension by fact
            if( alias(result, data) ) {
                map(result) += (this->compressedMatrix.transpose() * (map(data).template cast<Real>() * fact)).template cast<InReal>();
            } else {
                typename map_traits<InType>::map_type r = map(result);
                r.noalias() += (this->compressedMatrix.transpose() * (map(data).template cast<Real>() * fact)).template cast<InReal>();
            }
#endif
			
			return;
		}

		// convert the data to Eigen type
        VectorEigenOut aux1(this->rowSize()), aux2(this->colSize());

        for(size_t i = 0, n = data.size(); i < n; ++i) {
			for(unsigned j = 0; j < Nout; ++j) {
                aux1[Nout * i + j] = data[i][j];
			}
		}
		
		// compute the product
#if (SOFA_LINEARALGEBRA_HAVE_OPENMP == 1)
        aux2.noalias() = linearsolver::mul_EigenSparseDenseMatrix_MT( this->compressedMatrix.transpose(), aux1 );
#else
        aux2.noalias() = this->compressedMatrix.transpose() * aux1;
#endif

		// convert the result back to the Sofa type
        for(size_t i = 0, n = result.size(); i < n; ++i) {
			for(unsigned j = 0; j < Nin; ++j) {
                result[i][j] += aux2[Nin * i + j] * fact;
			}
		}
	}

	
public:

    /// compute result = A * data
    void mult( OutVecDeriv& result, const InVecDeriv& data ) const {
	    mult_impl(result, data);
    }

    /// compute result += A * data
    void addMult( OutVecDeriv& result, const InVecDeriv& data ) const {
	    addMult_impl(result, data, 1.0);
    }
      
    /// compute result += A * data * fact
    void addMult( OutVecDeriv& result, const InVecDeriv& data, const OutReal fact ) const {
        addMult_impl(result, data, fact);
    }

    /// compute result += A^T * data
    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const {
        addMultTranspose_impl(result, data, 1.0);
    }

    /// compute result += A^T * data * fact
    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data, const OutReal fact ) const {
        addMultTranspose_impl(result, data, fact);
    }

    static const std::string Name()
    {
        std::ostringstream o;
        o << "EigenMatrix";

        if constexpr (std::is_scalar<Real>::value)
        {
            if constexpr (std::is_same<float, Real>::value)
            {
                o << "f";
            }
            if constexpr (std::is_same<double, Real>::value)
            {
                o << "d";
            }
        }

        return o.str();
    }

private:
    //@{
    /** Auxiliary variables for methods beginBlockRow(unsigned row),
     * createBlock( unsigned column, const Block& b ) and
     * endBlockRow() */

    unsigned bRow;
    vector<unsigned> bColumns;
    vector<Block> blocks;
    //@}

	
	template<class T>
	bool canCast( const T& v ) const { 
		// is it contiguous ?
		typedef typename T::value_type value_type;
		typedef typename value_type::value_type scalar_type;
        return  (v.size() - 1) * sizeof(value_type) == (&v[v.size() - 1][0] - &v[0][0]) * sizeof(scalar_type); 
	}
};

} // namespace sofa::linearalgebra

namespace sofa
{

    /// Converts a BaseMatrix to a eigen sparse matrix encapsulated in a OwnershipSPtr.
    /// It the conversion needs to create a temporary matrix, it will be automatically deleted
    /// by the OwnershipSPtr (with ownership).
    /// It the conversion did not create a temporary data, and points to an existing matrix,
    /// the OwnershipSPtr does not take the ownership and won't delete anything.
    /// @TODO move this somewhere else?
    /// @author Matthieu Nesme
    template<class mat>
    helper::OwnershipSPtr<mat> convertSPtr( const linearalgebra::BaseMatrix* m) {
        assert( m );

        {
        typedef linearalgebra::EigenBaseSparseMatrix<SReal> matrixr;
        const matrixr* smr = dynamic_cast<const matrixr*> (m);
        // no need to create temporary data, so the SPtr does not take this ownership
        if ( smr ) return helper::OwnershipSPtr<mat>(&smr->compressedMatrix, false);
        }

        msg_warning("EigenSparseMatrix")<<"convertSPtr: slow matrix conversion (scalar type conversion)";

        {
        typedef linearalgebra::EigenBaseSparseMatrix<double> matrixd;
        const matrixd* smd = dynamic_cast<const matrixd*> (m);
        // the cast is creating a temporary matrix, the SPtr takes its ownership, so its deletion will be transparent
        if ( smd ) return helper::OwnershipSPtr<mat>( new mat(smd->compressedMatrix.cast<SReal>()), true );
        }

        {
        typedef linearalgebra::EigenBaseSparseMatrix<float> matrixf;
        const matrixf* smf = dynamic_cast<const matrixf*>(m);
        // the cast is creating a temporary matrix, the SPtr takes its ownership, so its deletion will be transparent
        if( smf ) return helper::OwnershipSPtr<mat>( new mat(smf->compressedMatrix.cast<SReal>()), true );
        }

        msg_warning("EigenSparseMatrix")<<"convertSPtr: very slow matrix conversion (from BaseMatrix)";

        mat* res = new mat(m->rowSize(), m->colSize());

        res->reserve(res->rows() * res->cols());
        for(unsigned i = 0, n = res->rows(); i < n; ++i) {
            res->startVec( i );
            for(unsigned j = 0, k = res->cols(); j < k; ++j) {
                SReal e = m->element(i, j);
                if( e ) res->insertBack(i, j) = e;
            }
        }

        // a temporary matrix is created, the SPtr takes its ownership, so its deletion will be transparent
        return helper::OwnershipSPtr<mat>(res, true);
    }
} // namespace sofa
