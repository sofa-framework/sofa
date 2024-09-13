struct NoInit;

explicit Matrix(const NoInit& noInit)
{}

void identity()
{
    *this = Identity();
}

void clear()
{
    *this = Zero();
}

auto ptr()
{
    return data();
}

auto ptr() const
{
    return data();
}

bool invert(const Matrix& mat)
{
    *this = mat.inverse();
    return true;
}

auto transposed() const
{
    return transpose();
}

auto& x()
{
    return this->operator()(0,0);
}

const auto& x() const
{
    return this->operator()(0,0);
}

auto& operator[](Index i)
{
    if constexpr (IsRowMajor && ColsAtCompileTime > 1)
    {
        return this->row(i);
    }
    else
    {
        return this->operator()(i);
    }
}

const auto& operator[](Index i) const
{
    if constexpr (IsRowMajor && ColsAtCompileTime > 1)
    {
        return this->row(i);
    }
    else
    {
        return this->operator()(i);
    }
}
