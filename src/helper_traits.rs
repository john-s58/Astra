use core::ops::{Add, Div, Mul, Rem, Sub};

pub trait Numeric:
    Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Rem<Output = Self>
    + Copy
    + PartialEq
    + PartialOrd
    + Default
    + std::ops::SubAssign
    + std::fmt::Debug
{
    fn from_i32(value: i32) -> Self;
    fn abs(&self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

impl Numeric for f32 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for f64 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for i8 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        i8::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for i16 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        i16::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for i32 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        i32::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for i64 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        i64::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for i128 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        i128::abs(*self)
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for u8 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        *self
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for u16 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        *self
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for u32 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        *self
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for u64 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        *self
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for u128 {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        *self
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}
impl Numeric for usize {
    fn from_i32(value: i32) -> Self {
        value as Self
    }
    fn abs(&self) -> Self {
        *self
    }
    fn zero() -> Self {
        0 as Self
    }
    fn one() -> Self {
        1 as Self
    }
}

pub trait Float: Numeric {}
impl Float for f32 {}
impl Float for f64 {}

pub trait Int: Numeric + Eq + Ord {}
impl Int for i8 {}
impl Int for i16 {}
impl Int for i32 {}
impl Int for i64 {}
impl Int for i128 {}
pub trait Unsigned: Numeric + Eq + Ord {}
impl Unsigned for u8 {}
impl Unsigned for u16 {}
impl Unsigned for u32 {}
impl Unsigned for u64 {}
impl Unsigned for u128 {}
impl Unsigned for usize {}
