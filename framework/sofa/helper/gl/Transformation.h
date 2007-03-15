/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_GL_TRANSFORMATION_H
#define SOFA_HELPER_GL_TRANSFORMATION_H

namespace sofa
{

namespace helper
{

namespace gl
{

class   		Transformation
{
public:

    double			translation[3];
    double			scale[3];
    double			rotation[4][4];

    double			objectCenter[3];

private:

public:

    Transformation();	// constructor
    ~Transformation();	// destructor



    Transformation&	operator=(const Transformation& transform);

    void			Apply();
    void			ApplyWithCentring();
    void			ApplyInverse();

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

private:void		InvertTransRotMatrix(double matrix[4][4]);
    void			InvertTransRotMatrix(double sMatrix[4][4],
            double dMatrix[4][4]);
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif // __TRANSFORMATION_H__
