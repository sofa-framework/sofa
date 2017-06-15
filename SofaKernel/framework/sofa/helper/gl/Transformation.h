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
#ifndef SOFA_HELPER_GL_TRANSFORMATION_H
#define SOFA_HELPER_GL_TRANSFORMATION_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace gl
{

class SOFA_HELPER_API Transformation
{
public:
    SReal			translation[3];
    SReal			scale[3];
    SReal			rotation[4][4];
    SReal			objectCenter[3];

public:
    Transformation();	// constructor
    ~Transformation();	// destructor
    Transformation&	operator=(const Transformation& transform);

    void Apply();
    void ApplyWithCentring();
    void ApplyInverse();

    template<class Vector>
    Vector operator*(Vector v) const
    {
        for(int c=0; c<3; c++)
            v[c] *= scale[c];
        Vector r;
        for(int c=0; c<3; c++)
            r[c] = rotation[0][c]*v[0]+rotation[1][c]*v[1]+rotation[2][c]*v[2];
        for(int c=0; c<3; c++)
            r[c] += translation[c];
        return r;
    }

private:
    void InvertTransRotMatrix(SReal matrix[4][4]);
    void InvertTransRotMatrix(SReal sMatrix[4][4],
            SReal dMatrix[4][4]);
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif // __TRANSFORMATION_H__
