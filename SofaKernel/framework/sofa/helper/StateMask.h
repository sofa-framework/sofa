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
#ifndef SOFA_HELPER_PARTICLEMASK_H
#define SOFA_HELPER_PARTICLEMASK_H
#include <sofa/helper/vector.h>

#include <sofa/defaulttype/Mat.h>
#include <Eigen/SparseCore>


namespace sofa
{

namespace helper
{

/**
 *  \brief Utility class to handle the mechanism of masks.
 *
 *  One of the most time-consuming process in Sofa is the transmission of forces and velocities through the mappings.
 *  If only a little subset of particles are used, we would like to propagate those forces (applyJT), and velocities (applyJ) to this subset only.
 *
 *  This class is used inside the BaseMechanicalState.
 *
 *  USAGE:
 *
 *     - Forcefields, Constraints
 *              which acts only on a little number of dofs should use the mask by only adding active entries
 *              in the fonction updateForceMask() (the default implementation adds every dofs in the mask)
 *
 *     - (Multi)Mappings
 *              they must propagate the mask from their child (tomodel) to their parents (frommodels)
 *              ApplyJ shoud check is the mask is active and use getEntry to check if a child dof is active
 *              ApplyJT shoud use getEntry to check if a child dof is active and CAN insert parent dofs in the parent mask.
 *              ApplyDJT, getJ/getJs shoud use getEntry to check if a child dof is active
 *              updateForceMask() must insert only active parent dofs in the parent mask (or should add nothing if parents have already been added in ApplyJT)
 *
 */
#


#ifdef SOFA_USE_MASK

class SOFA_HELPER_API StateMask
{

public:

    typedef helper::vector<bool> InternalStorage; // note this should be space-optimized (a bool = a bit) in the STL

    StateMask() : activated(false) {}

    /// filling-up (and resizing when necessary) the mask
    void assign( size_t size, bool value );

    /// the mask can be deactivated when the mappings must be applied to every dofs (e.g. propagatePosition)
    /// it must be activated when the mappings can be limited to active dofs
    void activate( bool a );
    inline bool isActivated() const { return activated; }

    /// add the given dof index in the mask
    inline void insertEntry( size_t index ) { mask[index]=true; }

    /// is the given dof index in the mask?
    /// @warning always returns the mask value w/o checking if the mask is activated (ie do no forget to check if mask is activated in Mapping::applyJ)
    inline bool getEntry( size_t index ) const { return mask[index]; } // unsafe to be use where we do not care if the mapping in deactivated

    /// getting mask entries is useful for advanced uses.
    const InternalStorage& getEntries() const { return mask; }

    void resize( size_t size );
    inline void clear() { mask.clear(); }
    size_t size() const { return mask.size(); }

    inline friend std::ostream& operator<< ( std::ostream& os, const StateMask& sm )
    {
        return os << sm.mask;
    }


    /// filtering the given input matrix by using the mask as a diagonal projection matrix
    /// output = mask.asDiagonal() * input
    template<class Real>
    void maskedMatrix( Eigen::SparseMatrix<Real,Eigen::RowMajor>& output, const Eigen::SparseMatrix<Real,Eigen::RowMajor>& input, size_t blockSize=1 ) const
    {
        typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> Mat;

        output.resize( input.rows(), input.cols() );

        for( size_t k=0 ; k<mask.size() ; ++k )
        {
            for( size_t i=0 ; i<blockSize ; ++i )
            {
                int row = k*blockSize+i;
                output.startVec( row );
                if( mask[k] )
                {
                    for( typename Mat::InnerIterator it(input,row) ; it ; ++it )
                        output.insertBack( row, it.col() ) = it.value();
                }
            }
        }
        output.finalize();
    }



    /// return the number of dofs in the mask
    size_t nbActiveDofs() const;

    /// return a hash, useful to check if the mask changed
    size_t getHash() const;

protected:

    InternalStorage mask; // note this should be space-optimized (a bool = a bit) in the STL
    bool activated; // automatic switch (the mask is only used for specific operations)

};

#else

class SOFA_HELPER_API StateMask
{

public:

    StateMask() : m_size(0) {}

    /// filling-up (and resizing when necessary) the mask
    void assign( size_t size, bool ) { m_size=size; }

    /// the mask can be deactivated when the mappings must be applied to every dofs (e.g. propagatePosition)
    /// it must be activated when the mappings can be limited to active dofs
    void activate( bool ) {}
    inline bool isActivated() const { return false; }

    /// add the given dof index in the mask
    inline void insertEntry( size_t /*index */) {}

    /// is the given dof index in the mask?
    /// @warning always returns the mask value w/o checking if the mask is activated (ie do no forget to check if mask is activated in Mapping::applyJ)
    inline bool getEntry( size_t /*index*/ ) const {
        return true;
    } // unsafe to be use where we do not care if the mapping in deactivated

    /// getting mask entries is useful for advanced uses.
    //    const InternalStorage& getEntries() const { return mask; }

    void resize( size_t size ) { m_size=size; }
    inline void clear() { m_size=0; }

    size_t size() const {
        return m_size;
    }

    inline friend std::ostream& operator<< ( std::ostream& os, const StateMask& /*sm*/ ) { return os; }


    /// filtering the given input matrix by using the mask as a diagonal projection matrix
    /// output = mask.asDiagonal() * input
    template<class Real>
    void maskedMatrix( Eigen::SparseMatrix<Real,Eigen::RowMajor>& output, const Eigen::SparseMatrix<Real,Eigen::RowMajor>& input, size_t blockSize=1 ) const {output=input;}

    /// return the number of dofs in the mask
    size_t nbActiveDofs() const {return m_size;}

    size_t getHash() const { return 0; }

protected:

    size_t m_size;

};

#endif



} // namespace helper

} // namespace sofa

#endif
