#ifndef CUDAMATH_H
#define CUDAMATH_H

__device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ void operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ void operator-=(float3& a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ void operator*=(float3& a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__device__ float3 operator*(float a, float3 b)
{
    return make_float3(a*b.x, a*b.y, a*b.z);
}


__device__ float3 mul(float3 a, float3 b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ float dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


#endif
