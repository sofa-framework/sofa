

__kernel void FixedConstraint3t_projectResponseIndexed(
    int size,
    __global const int* indices,
    __global Real* dx
)
{
    int index =get_global_id(0);
    int indice = 3*indices[index];
    if (index < size)
    {
        dx[indice+0] = 0.0f;
        dx[indice+1] = 0.0f;
        dx[indice+2] = 0.0f;
    }
}
