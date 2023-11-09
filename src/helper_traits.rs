pub trait Float {}
impl Float for f32 {}
impl Float for f64 {}
pub trait Int {}
impl Int for i8 {}
impl Int for i16 {}
impl Int for i32 {}
impl Int for i64 {}
impl Int for i128 {}
pub trait Unsigned {}
impl Unsigned for u8 {}
impl Unsigned for u16 {}
impl Unsigned for u32 {}
impl Unsigned for u64 {}
impl Unsigned for u128 {}
impl Unsigned for usize {}
pub trait Numeric {}
impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}
impl Numeric for usize {}

// trait NewTrait: Clone + Default + OtherTraits {}
// impl<T> NewTrait for T where T: Clone + Default + OtherTraits {}