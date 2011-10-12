/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_STATE_H
#define SOFA_CORE_STATE_H

#include <sofa/core/BaseState.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <limits>

namespace sofa
{

namespace core
{

/**
 *  \brief Component storing position and velocity vectors.
 *
 *  This class define the interface of components used as source and
 *  destination of regular (non mechanical) mapping. It is then specialized as
 *  MechanicalState (storing other mechanical data) or MappedModel (if no
 *  mechanical data is used, such as for VisualModel).
 *
 *  The given DataTypes class should define the following internal types:
 *  \li \code Real \endcode : scalar values (float or double).
 *  \li \code Coord \endcode : position values.
 *  \li \code Deriv \endcode : derivative values (velocity).
 *  \li \code VecReal \endcode : container of scalar values with the same API as sofa::helper::vector.
 *  \li \code VecCoord \endcode : container of Coord values with the same API as sofa::helper::vector.
 *  \li \code VecDeriv \endcode : container of Deriv values with the same API as sofa::helper::vector.
 *  \li \code MatrixDeriv \endcode : vector of Jacobians (sparse constraint matrices).
 *
 */
template<class TDataTypes>
class State : public virtual BaseState
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(State,TDataTypes), BaseState);

    typedef TDataTypes DataTypes;
    /// Scalar values (float or double).
    typedef typename DataTypes::Real Real;
    /// Position values.
    typedef typename DataTypes::Coord Coord;
    /// Derivative values (velocity, forces, displacements).
    typedef typename DataTypes::Deriv Deriv;
    /// Container of scalar values with the same API as sofa::helper::vector.
    typedef typename DataTypes::VecReal VecReal;
    /// Container of Coord values with the same API as sofa::helper::vector.
    typedef typename DataTypes::VecCoord VecCoord;
    /// Container of Deriv values with the same API as sofa::helper::vector.
    typedef typename DataTypes::VecDeriv VecDeriv;
    /// Vector of Jacobians (sparse constraint matrices).
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
protected:
    virtual ~State() { }
public:
    /// @name New vectors access API based on VecId
    /// @{

    virtual Data< VecCoord >* write(VecCoordId v) = 0;
    virtual const Data< VecCoord >* read(ConstVecCoordId v) const = 0;

    virtual Data< VecDeriv >* write(VecDerivId v) = 0;
    virtual const Data< VecDeriv >* read(ConstVecDerivId v) const = 0;

    virtual Data< MatrixDeriv >* write(MatrixDerivId v) = 0;
    virtual const Data< MatrixDeriv >* read(ConstMatrixDerivId v) const = 0;

    /// @}

    /// @name Old specific vectors access API (now limited to read-only accesses)
    /// @{

    /// Return the current position vector.
    /// @deprecated use read(ConstVecCoordId::position()) instead.
    virtual const VecCoord* getX()  const
    {
        const Data<VecCoord>* v = read(ConstVecCoordId::position());
        return (v == NULL) ? NULL : &(v->getValue());
    }

    /// Return the current velocity vector.
    /// @deprecated use read(ConstVecDerivId::velocity()) instead.
    virtual const VecDeriv* getV()  const
    {
        const Data<VecDeriv>* v = read(ConstVecDerivId::velocity());
        return (v == NULL) ? NULL : &(v->getValue());
    }

    /// Return the current rest position vector
    /// (return NULL if the state does not store rest position).
    /// @deprecated use read(ConstVecCoordId::restPosition()) instead.
    virtual const VecCoord* getX0() const
    {
        const Data<VecCoord>* v = read(ConstVecCoordId::restPosition());
        return (v == NULL) ? NULL : &(v->getValue());
    }

    /// Return the current normal vector
    /// (return NULL if the state does not store normal).
    /// @deprecated use read(ConstVecDerivId::velocity()) instead.
    virtual const VecDeriv* getN() const
    {
        const Data<VecDeriv>* v = read(ConstVecDerivId::normal());
        return (v == NULL) ? NULL : &(v->getValue());
    }

    /// @}

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const State<DataTypes>* = NULL)
    {
        return TDataTypes::Name();
    }

    virtual void computeBBox(const core::ExecParams* params)
    {
        const VecCoord& x = read(ConstVecCoordId::position())->getValue(params);
        const unsigned int xSize = x.size();

        if (xSize <= 0)
            return;

        const Real max_real = std::numeric_limits<Real>::max();
        const Real min_real = std::numeric_limits<Real>::min();
        Real p[3] = {0,0,0};
        Real maxBBox[3] = {min_real,min_real,min_real};
        Real minBBox[3] = {max_real,max_real,max_real};

        for (unsigned int i = 0; i < xSize; i++)
        {
            DataTypes::get(p[0], p[1], p[2], x[i]);
            for (int c = 0; c < 3; c++)
            {
                if (p[c] > maxBBox[c])
                    maxBBox[c] = p[c];

                if (p[c] < minBBox[c])
                    minBBox[c] = p[c];
            }
        }

        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
    }
};

} // namespace core

} // namespace sofa

#endif
