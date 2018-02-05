/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/helper/gl/Transformation.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace helper
{

namespace gl
{

// --------------------------------------------------------------------------------------
// --- Constructor
// --------------------------------------------------------------------------------------
Transformation::Transformation()
{
    /*register*/ int	i, j;

    for (i = 0; i < 3; i++)
    {
        translation[i] = 0.0;
        scale[i] = 1.0;
        objectCenter[i] = 0.0;
    }

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            rotation[i][j] = 0.0;
        }
        rotation[i][i] = 1.0;
    }
}


// --------------------------------------------------------------------------------------
// --- Destructor
// --------------------------------------------------------------------------------------
Transformation::~Transformation()
{
}


// --------------------------------------------------------------------------------------
// --- Operator =
// --------------------------------------------------------------------------------------
Transformation& Transformation::operator=(const Transformation& transform)
{
    /*register*/ int i, j;

    for (i = 0; i < 3; i++)
    {
        translation[i] = transform.translation[i];
        scale[i] = transform.scale[i];
        objectCenter[i] = transform.objectCenter[i];
    }

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            rotation[i][j] = transform.rotation[i][j];
        }
    }

    return *this;
}


// --------------------------------------------------------------------------------------
// --- Apply the transformation
// --------------------------------------------------------------------------------------
void Transformation::Apply()
{
#ifndef SOFA_NO_OPENGL
    helper::gl::glTranslate(translation[0], translation[1], translation[2]);
    helper::gl::glMultMatrix((SReal *)rotation);
    helper::gl::glScale(scale[0], scale[1], scale[2]);
#endif /* SOFA_NO_OPENGL */
}


// --------------------------------------------------------------------------------------
// --- First center the object, then apply the transformation (to align with the corresponding texture)
// --------------------------------------------------------------------------------------
void Transformation::ApplyWithCentring()
{
    Apply();

#ifndef SOFA_NO_OPENGL
    helper::gl::glTranslate(-objectCenter[0], -objectCenter[1], -objectCenter[2]);
#endif /* SOFA_NO_OPENGL */
}


// --------------------------------------------------------------------------------------
// --- Apply the inverse transformation
// --------------------------------------------------------------------------------------
void Transformation::ApplyInverse()
{
    SReal	iRotation[4][4];

    InvertTransRotMatrix(rotation, iRotation);

#ifndef SOFA_NO_OPENGL
    helper::gl::glScale((SReal)1.0 / scale[0], (SReal)1.0 / scale[1], (SReal)1.0 / scale[2]);
    helper::gl::glMultMatrix((SReal *)rotation);
    helper::gl::glTranslate(-translation[0], -translation[1], -translation[2]);
#endif /* SOFA_NO_OPENGL */
}


//----------------------------------------------------------------------------
//--- Inversion for 4x4 matrix only containing rotations and translations
//--- Transpose rotation matrix and mutiple by -1 translation row
//----------------------------------------------------------------------------
void Transformation::InvertTransRotMatrix(SReal matrix[4][4])
{
    SReal	tmp;

    tmp = matrix[0][1];
    matrix[0][1] = matrix[1][0];
    matrix[1][0] = tmp;

    tmp = matrix[0][2];
    matrix[0][2] = matrix[2][0];
    matrix[2][0] = tmp;

    tmp = matrix[1][2];
    matrix[1][2] = matrix[2][1];
    matrix[2][1] = tmp;

    matrix[3][0] = -matrix[3][0];
    matrix[3][1] = -matrix[3][1];
    matrix[3][2] = -matrix[3][2];
}


//----------------------------------------------------------------------------
//--- Invert the source matrix sMatrix and put the result in the destination matrix dMatrix
//--- Inversion for 4x4 matrix only containing rotations and translations
//--- Transpose rotation matrix and mutiple by -1 translation row
//----------------------------------------------------------------------------
void Transformation::InvertTransRotMatrix(SReal sMatrix[4][4],
        SReal dMatrix[4][4])
{
    /*register*/ int	i, j;

    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            dMatrix[i][j] = sMatrix[i][j];
        }
    }
    InvertTransRotMatrix(dMatrix);
}

} // namespace gl

} // namespace helper

} // namespace sofa

