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
#ifndef SOFA_CORE_STATE_INL
#define SOFA_CORE_STATE_INL

#include <sofa/core/State.h>

namespace sofa
{

namespace core
{

template<class DataTypes>
objectmodel::BaseData* State<DataTypes>::baseWrite(VecId v)
{
    switch (v.getType())
    {
    case V_NULL: return NULL;
    case V_COORD: return write(VecCoordId(v));
    case V_DERIV: return write(VecDerivId(v));
    case V_MATDERIV: return write(MatrixDerivId(v));
    }
}

template<class DataTypes>
const objectmodel::BaseData* State<DataTypes>::baseRead(ConstVecId v) const
{
    switch (v.getType())
    {
    case V_NULL: return NULL;
    case V_COORD: return read(ConstVecCoordId(v));
    case V_DERIV: return read(ConstVecDerivId(v));
    case V_MATDERIV: return read(ConstMatrixDerivId(v));
    }
}

template<class DataTypes>
void State<DataTypes>::computeBBox(const core::ExecParams* params)
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

} // namespace core

} // namespace sofa

#endif
