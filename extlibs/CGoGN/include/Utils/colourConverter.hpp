/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
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
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

namespace CGoGN
{

namespace Utils
{

template<typename REAL>
ColourConverter<REAL>::ColourConverter(const VEC3& col, const enum ColourEncoding& enc) :
	RGB(NULL),
	Luv(NULL),
	Lab(NULL),
	HSV(NULL),
	XYZ(NULL),
	HSL(NULL)
{
	originalEnc = enc ;

	switch(originalEnc)
	{
		case(C_RGB):
			#ifdef DEBUG
				if (!(-0.001 < col[0] && col[0] < 1.001 && -0.001 < col[1] && col[1] < 1.001 &&  -0.001 < col[2] && col[2] < 1.001))
					std::cerr << "Warning : an unvalid RGB color was entered in ColourConverter constructor" << std::endl ;
			#endif
			RGB = new VEC3(col) ;
			break ;

		case (C_Luv) :
			#ifdef DEBUG
				if (!(-0.001 < col[0] && col[0] < 100.001 && -83.001 < col[1] && col[1] < 175.001 &&  -134.001 < col[2] && col[2] < 108.001))
					std::cerr << "Warning : an unvalid Luv color was entered in ColourConverter constructor" << std::endl ;
			#endif
			Luv = new VEC3(col) ;
			break ;

		case (C_XYZ) :
			#ifdef DEBUG
				if (!(-0.001 < col[0] && col[0] < 1.001 && -0.001 < col[1] && col[1] < 1.001 &&  -0.001 < col[2] && col[2] < 1.001))
					std::cerr << "Warning : an unvalid XYZ color was entered in ColourConverter constructor" << std::endl ;
			#endif
			XYZ = new VEC3(col) ;
			break ;

		case (C_Lab) :
			#ifdef DEBUG
				if (!(-0.001 < col[0] && col[0] < 100.001 && -86.001 < col[1] && col[1] < 98.001 && -108.001 < col[2] && col[2] < 95.001))
						std::cerr << "Warning : an unvalid Lab color was entered in ColourConverter constructor" << std::endl ;
			#endif
			Lab = new VEC3(col) ;
			break ;
		case (C_HSV) :
			#ifdef DEBUG
				if (!(-0.001 < col[0] && col[0] < 1.001 && -0.001 < col[1] && col[1] < 1.001 && -0.001 < col[2] && col[2] < 1.001))
						std::cerr << "Warning : an unvalid HSV color was entered in ColourConverter constructor" << std::endl ;
			#endif
			HSV = new VEC3(col) ;
			break ;
		case (C_HSL) :
			#ifdef DEBUG
				if (!(-0.001 < col[0] && col[0] < 1.001 && -0.001 < col[1] && col[1] < 1.001 && -0.001 < col[2] && col[2] < 1.001))
						std::cerr << "Warning : an unvalid HSL color was entered in ColourConverter constructor" << std::endl ;
			#endif
			HSL = new VEC3(col) ;
			break ;
	}
}

template<typename REAL>
ColourConverter<REAL>::~ColourConverter()
{
	delete RGB ;
	delete Luv ;
	delete XYZ ;
	delete Lab ;
	delete HSV ;
	delete HSL ;
}

template<typename REAL>
Geom::Vector<3,REAL> ColourConverter<REAL>::getColour(enum ColourEncoding enc)
{
	switch (enc)
	{
	case (C_RGB) :
		return getRGB() ;
		break ;

	case (C_XYZ) :
		return getXYZ() ;
		break ;

	case (C_Luv) :
		return getLuv() ;
		break ;

	case (C_Lab) :
		return getLab() ;
		break ;

	case (C_HSV) :
		return getHSV() ;
		break ;

	case (C_HSL) :
		return getHSL() ;
		break ;

	default :
		assert(!"Should never arrive here : ColourConverter::getColour default case") ;
		return getOriginal() ;
	}
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getOriginal()
{
	return getColour(this->originalEnc) ;
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getRGB()
{
	if (RGB == NULL)
		convert(originalEnc,C_RGB) ;

	return *RGB ;
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getLuv()
{
	if (Luv == NULL)
		convert(originalEnc,C_Luv) ;

	return *Luv ;
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getLab()
{
	if (Lab == NULL)
		convert(originalEnc,C_Lab) ;

	return *Lab ;
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getXYZ()
{
	if (XYZ == NULL) {
		convert(originalEnc,C_XYZ) ;
	}

	return *XYZ ;
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getHSV()
{
	if (HSV == NULL)
		convert(originalEnc,C_HSV) ;

	return *HSV ;
}

template<typename REAL>
Geom::Vector<3,REAL>
ColourConverter<REAL>::getHSL()
{
	if (HSL == NULL)
		convert(originalEnc,C_HSL) ;

	return *HSL ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertRGBtoXYZ()
{
	Geom::Matrix<3,3,REAL> M ;

	M(0,0) = 0.412453 ;
	M(0,1) = 0.357580 ;
	M(0,2) = 0.180423 ;

	M(1,0) = 0.212671 ;
	M(1,1) = 0.715160 ;
	M(1,2) = 0.072169 ;

	M(2,0) = 0.019334 ;
	M(2,1) = 0.119193 ;
	M(2,2) = 0.950227 ;

	VEC3 c = M * (*RGB) ;

	if (XYZ != NULL)
		*XYZ = c ;
	else
		XYZ = new VEC3(c) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertXYZtoRGB()
{
	Geom::Matrix<3,3,REAL> M ;

	M(0,0) = 3.240479 ;
	M(0,1) = -1.537150 ;
	M(0,2) = -0.498535 ;

	M(1,0) = -0.969256 ;
	M(1,1) = 1.875992 ;
	M(1,2) = 0.041556 ;

	M(2,0) = 0.055648 ;
	M(2,1) = -0.204043 ;
	M(2,2) = 1.057311 ;

	VEC3 c = M * (*XYZ) ;

	if (RGB != NULL)
		*RGB = c ;
	else
		RGB = new VEC3(c) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertRGBtoHSV()
{
	const REAL& r = (*RGB)[0] ;
	const REAL& g = (*RGB)[1] ;
	const REAL& b = (*RGB)[2] ;
	const REAL& max = std::max(std::max(r,g),b) ;
	const REAL& min = std::min(std::min(r,g),b) ;


	VEC3 c ;
	REAL& H = c[0] ;
	REAL& S = c[1] ;
	REAL& V = c[2] ;

	const REAL diff = max - min ;

	V = max ;
	S = max == 0. ? 0 : diff / max ;


	if (max == min)
	{
		H = 0 ;
	}
	else
	{
		if (max == r)
		{
			H = (g - b) / diff + (g < b ? 6 : 0) ;
		}
		else if (max == g)
		{
			H = (b - r) / diff + 2 ;
		}
		else if (max == b)
		{
			H = (r - g) / diff + 4 ;
		}
	}
	H /= 6. ;

	if (HSV != NULL)
		*HSV = c ;
	else
		HSV = new VEC3(c) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertHSVtoRGB()
{
	const REAL& H = (*HSV)[0] ;
	const REAL& S = (*HSV)[1] ;
	const REAL& V = (*HSV)[2] ;

	const int i = std::floor(H * 6);
	const REAL f = H * 6 - i;
	const REAL p = V * (1 - S);
	const REAL q = V * (1 - f * S);
	const REAL t = V * (1 - (1 - f) * S);

	VEC3 c ;
	REAL& r = c[0] ;
	REAL& g = c[1] ;
	REAL& b = c[2] ;

	switch(i % 6)
	{
	case 0: r = V, g = t, b = p; break;
	case 1: r = q, g = V, b = p; break;
	case 2: r = p, g = V, b = t; break;
	case 3: r = p, g = q, b = V; break;
	case 4: r = t, g = p, b = V; break;
	case 5: r = V, g = p, b = q; break;
	}

	if (RGB != NULL)
		*RGB = c ;
	else
		RGB = new VEC3(c) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertRGBtoHSL()
{
	const REAL& r = (*RGB)[0] ;
	const REAL& g = (*RGB)[1] ;
	const REAL& b = (*RGB)[2] ;
	const REAL& max = std::max(std::max(r,g),b) ;
	const REAL& min = std::min(std::min(r,g),b) ;

	VEC3 c ;
	REAL& H = c[0] ;
	REAL& S = c[1] ;
	REAL& L = c[2] ;

	const REAL sum = max + min ;
	L = sum / 2. ;

	if (max == min)
	{
		H = 0 ;
		S = 0 ;
	}
	else
	{
		const REAL diff = max - min ;
		S = L > 0.5 ? diff / (2 - sum) : diff / sum ;

		if (max == r)
		{
			H = (g - b) / diff + (g < b ? 6 : 0) ;
		}
		else if (max == g)
		{
			H = (b - r) / diff + 2 ;
		}
		else if (max == b)
		{
			H = (r - g) / diff + 4 ;
		}
	}
	H /= 6. ;

	if (HSL != NULL)
		*HSL = c ;
	else
		HSL = new VEC3(c) ;
}

template<typename REAL>
REAL
ColourConverter<REAL>::hue2rgb(const REAL& p, const REAL& q, REAL t)
{
	if(t < 0)
		t += 1 ;
	if(t > 1)
		t -= 1 ;
	if(t < 1/6.)
		return p + (q - p) * 6 * t ;
	if(t < 1/2.)
		return q ;
	if(t < 2/3.)
		return p + (q - p) * (2/3. - t) * 6 ;

	return p ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertHSLtoRGB()
{
	const REAL& H = (*HSL)[0] ;
	const REAL& S = (*HSL)[1] ;
	const REAL& L = (*HSL)[2] ;

	VEC3 c ;
	REAL& r = c[0] ;
	REAL& g = c[1] ;
	REAL& b = c[2] ;

    if(S < 1e-8)
    {
    	r = L ;
    	g = L ;
    	b = L ;
    }
    else
    {
    	const REAL q = L < 0.5 ? L * (1 + S) : L + S - L * S;
    	const REAL p = 2 * L - q ;
    	r = hue2rgb(p, q, H + 1/3.) ;
    	g = hue2rgb(p, q, H) ;
    	b = hue2rgb(p, q, H - 1/3.) ;
    }

	if (RGB != NULL)
		*RGB = c ;
	else
		RGB = new VEC3(c) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertXYZtoLuv()
{
	REAL L,u,v ;

	REAL &X = (*XYZ)[0] ;
	REAL &Y = (*XYZ)[1] ;
	REAL &Z = (*XYZ)[2] ;

	REAL Ydiv = Y/Yn ;
	if (Ydiv > 0.008856)
		L = 116.0 * pow(Ydiv,1.0/3.0) - 16.0 ;
	else // near black
		L = 903.3 * Ydiv ;

	REAL den = X + 15.0 * Y + 3 * Z ;
	REAL u1 = (4.0 * X) / den ;
	REAL v1 = (9.0 * Y) / den ;
	u = 13*L * (u1 - un) ;
	v = 13*L * (v1 - vn) ;

	if (Luv != NULL)
		*Luv = VEC3(L,u,v) ;
	else
		Luv = new VEC3(L,u,v) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertLuvToXYZ()
{
	REAL X,Y,Z ;

	REAL &L = (*Luv)[0] ;
	REAL &u = (*Luv)[1] ;
	REAL &v = (*Luv)[2] ;

	if (L > 8.0)
		Y = pow(((L+16.0) / 116.0),3) ;
	else // near black
		Y = Yn * L / 903.3 ;

	REAL den = 13.0 * L ;
	REAL u1 = u/den + un ;
	REAL v1 = v/den + vn ;
	den = 4.0*v1 ;
	X = Y * 9.0 * u1 / den ;
	Z = Y * (12.0 - 3.0*u1 - 20.0*v1) / den ;

	if (XYZ != NULL)
		*XYZ = VEC3(X,Y,Z) ;
	else
		XYZ = new VEC3(X,Y,Z) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertXYZtoLab()
{
	REAL L,a,b ;

	REAL &X = (*XYZ)[0] ;
	REAL &Y = (*XYZ)[1] ;
	REAL &Z = (*XYZ)[2] ;

	struct Local
	{
		static REAL f(REAL x)
		{
			if (x > 0.008856)
			return pow(x,1.0/3.0) ;
			else
				return 7.787 * x + 16.0/116.0 ;
		}
	} ;

	if (Y > 0.008856)
		L = 116.0f * pow(Y,1.0f/3.0) - 16 ;
	else // near black
		L = 903.3 * Y ;

	a = 500.0 * (Local::f(X/Xn) - Local::f(Y/Yn)) ;
	b = 200.0 * (Local::f(Y/Yn) - Local::f(Z/Zn)) ;

	if (Lab != NULL)
		*Lab = VEC3(L,a,b) ;
	else
		Lab = new VEC3(L,a,b) ;
}

template<typename REAL>
void
ColourConverter<REAL>::convertLabToXYZ()
{
	REAL X,Y,Z ;

	REAL &L = (*Lab)[0] ;
	REAL &a = (*Lab)[1] ;
	REAL &b = (*Lab)[2] ;

	struct Local
	{
		static REAL f(REAL x)
		{
			if (x > 0.206893)
			return pow(x,3.0) ;
			else
				return x / 7.787 - 16.0/903.3 ;
		}
	} ;

	if (L > 8.0)
		Y = pow(((L+16.0) / 116.0),3) ;
	else // near black
		Y = L / 903.3 ;

	REAL nom = (L+16.0) / 116.0 ;
	X = Xn * Local::f( nom +  a/500.0) ;
	Z = Zn * Local::f( nom -  b/200.0) ;

	if (XYZ != NULL)
		*XYZ = VEC3(X,Y,Z) ;
	else
		XYZ = new VEC3(X,Y,Z) ;
}

template<typename REAL>
bool
ColourConverter<REAL>::convert(enum ColourEncoding from, enum ColourEncoding to)
{
	if (to == from)
	{
		#ifdef DEBUG
			std::cout << "WARNING ColourConverter::convert(from,to) : conversion into same colour space" << std::endl ;
		#endif
		return true ;
	}

	switch(from)
	{
		case(C_RGB) :
			switch (to)
			{
				case (C_XYZ) :
					if (XYZ == NULL)
						convertRGBtoXYZ() ;
					break ;
				case (C_Luv) :
					if (Luv == NULL)
					{
						if (XYZ == NULL)
							convertRGBtoXYZ() ;
						convertXYZtoLuv() ;
					}
					break ;
				case(C_Lab) :
					if (Lab == NULL)
					{
						if (XYZ == NULL)
							convertRGBtoXYZ() ;
						convertXYZtoLab() ;
					}
				break ;
				case(C_HSV) :
					if (HSV == NULL)
						convertRGBtoHSV() ;
					break ;
				case(C_HSL) :
					if (HSL == NULL)
						convertRGBtoHSL() ;
					break ;
				default :
					std::cerr << "Colour conversion not supported" << std::endl ;
					return false ;
			}
			break ;

		case(C_Luv) :
			switch(to)
			{
				case(C_RGB) : {
					if (RGB == NULL)
					{
						if (XYZ == NULL)
							convertLuvToXYZ() ;
						convertXYZtoRGB() ;
					}
					break ;
				}
				case(C_XYZ) : {
					if (XYZ == NULL)
						convertLuvToXYZ() ;
					break ;
				}
				case(C_Lab) :
					if (Lab == NULL)
					{
						if (XYZ == NULL)
							convertLuvToXYZ() ;
						convertXYZtoLab() ;
					}
					break ;
				case(C_HSV) :
					if (HSV == NULL)
					{
						if (RGB == NULL)
						{
							if (XYZ == NULL)
								convertLuvToXYZ() ;
							convertXYZtoRGB() ;
						}
						convertRGBtoHSV() ;
					}
					break ;
				case(C_HSL) : {
					if (HSL == NULL)
					{
						if (RGB == NULL)
						{
							if (XYZ == NULL)
								convertLuvToXYZ() ;
							convertXYZtoRGB() ;
						}
						convertRGBtoHSL() ;
					}
					break ;
				}
				default :
					std::cerr << "Colour conversion not supported" << std::endl ;
					return false ;
			}
			break ;

		case(C_XYZ) :
			switch (to)
			{
				case(C_RGB) :
					if (RGB == NULL)
						convertXYZtoRGB() ;
					break ;
				case(C_Luv) :
					if (Luv == NULL)
						convertXYZtoLuv() ;
					break ;
				case(C_Lab) :
					if (Lab == NULL)
						convertXYZtoLab() ;
					break ;
				case(C_HSV) :
						if (HSV == NULL)
						{
							if (RGB == NULL)
								convertXYZtoRGB() ;
							convertRGBtoHSV() ;
						}
					break ;
				case(C_HSL) :
					if (HSL == NULL)
					{
						if (RGB == NULL)
							convertXYZtoRGB() ;
						convertRGBtoHSL() ;
					}
					break ;
				default :
					std::cerr << "Colour conversion not supported" << std::endl ;
					return false ;
			}
			break ;

		case(C_Lab) :
			switch (to)
			{
				case(C_RGB) : {
					if (RGB == NULL)
					{
						if (XYZ == NULL)
							convertLabToXYZ() ;
						convertXYZtoRGB() ;
					}
					break ;
				}
				case(C_XYZ) : {
					if (XYZ == NULL)
						convertLabToXYZ() ;
					break ;
				}
				case(C_Luv) :
					if (Luv == NULL)
					{
						if (XYZ == NULL)
							convertLabToXYZ() ;
						convertXYZtoLuv() ;
					}
					break ;
				case(C_HSV) : {
					if (HSV == NULL)
					{
						if (RGB == NULL)
						{
							if (XYZ == NULL)
								convertLabToXYZ() ;
							convertXYZtoRGB() ;
						}
						convertRGBtoHSV() ;
					}
					break ;
				}
				case(C_HSL) : {
					if (HSL == NULL)
					{
						if (RGB == NULL)
						{
							if (XYZ == NULL)
								convertLabToXYZ() ;
							convertXYZtoRGB() ;
						}
						convertRGBtoHSL() ;
					}
					break ;
				}
				default :
					std::cerr << "Colour conversion not supported" << std::endl ;
					return false ;
			}
			break ;

		case(C_HSV) :
				switch (to)
				{
					case(C_RGB) : {
						if (RGB == NULL)
						{
							convertHSVtoRGB() ;
						}
						break ;
					}
					case(C_XYZ) : {
						if (XYZ == NULL)
						{
							if (RGB == NULL)
								convertHSVtoRGB() ;
							convertRGBtoXYZ() ;
						}
						break ;
					}
					case(C_Lab) :
						if (Lab == NULL)
						{
							if (XYZ == NULL)
							{
								if (RGB == NULL)
								{
									convertHSVtoRGB() ;
								}
								convertRGBtoXYZ() ;
							}
							convertXYZtoLab() ;
						}
						break ;
					case(C_Luv) :
						if (Luv == NULL)
						{
							if (XYZ == NULL)
							{
								if (RGB == NULL)
								{
									convertHSVtoRGB() ;
								}
								convertRGBtoXYZ() ;
							}
							convertXYZtoLuv() ;
						}
						break ;
					case(C_HSL) :
						if (HSL == NULL)
						{
							if (RGB == NULL)
								convertHSVtoRGB() ;
							convertRGBtoHSL() ;
						}
						break ;
					default :
						std::cerr << "Colour conversion not supported" << std::endl ;
						return false ;
				}
				break ;

				case(C_HSL) :
						switch (to)
						{
							case(C_RGB) : {
								if (RGB == NULL)
								{
									convertHSLtoRGB() ;
								}
								break ;
							}
							case(C_XYZ) : {
								if (XYZ == NULL)
								{
									if (RGB == NULL)
										convertHSLtoRGB() ;
									convertRGBtoXYZ() ;
								}
								break ;
							}
							case(C_Lab) :
								if (Lab == NULL)
								{
									if (XYZ == NULL)
									{
										if (RGB == NULL)
										{
											convertHSLtoRGB() ;
										}
										convertRGBtoXYZ() ;
									}
									convertXYZtoLab() ;
								}
								break ;
							case(C_Luv) :
								if (Luv == NULL)
								{
									if (XYZ == NULL)
									{
										if (RGB == NULL)
										{
											convertHSLtoRGB() ;
										}
										convertRGBtoXYZ() ;
									}
									convertXYZtoLuv() ;
								}
								break ;
							case(C_HSL) :
								if (HSL == NULL)
								{
									if (RGB == NULL)
										convertHSLtoRGB() ;
									convertRGBtoHSV() ;
								}
								break ;
							default :
								std::cerr << "Colour conversion not supported" << std::endl ;
								return false ;
						}
						break ;

		default :
			std::cerr << "Colour conversion not supported" << std::endl ;
			return false ;
	}

	return true ;
}

} // namespace Utils

} // namespace CGoGN


