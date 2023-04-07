pub fn new() -> Self

pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self

pub fn from_element(element: f64, shape: Vec<usize>) -> Self

pub fn from_fn(shape: Vec<usize>, func: impl Fn() -> f64) -> Self

pub fn reshape(self, new_shape: Vec<usize>) -> Option<Self>

pub fn identity(shape: Vec<usize>) -> Option<Self>

pub fn matrix(m: usize, n: usize, data: Vec<f64>) -> Option<Self>

pub fn zeros(shape: &[usize]) -> Self

pub fn transpose(&self) -> Self

pub fn get_index(&self, indices: &[usize]) -> usize

pub fn get_indices(&self, index: usize) -> Option<Vec<usize>>

pub fn get_element(&self, indices: &[usize]) -> Option<&f64>

pub fn get_element_mut(&mut self, indices: &[usize]) -> Option<&mut f64>

pub fn set_element(&mut self, indices: &[usize], value: f64)

pub fn map(self, fun: impl Fn(f64) -> f64) -> Self

pub fn sum(&self) -> f64

pub fn dot(&self, other: &Self) -> Option<Self>

pub fn lu_decomposition(&self) -> Option<(Self, Self)>

pub fn norm(&self) -> Result<f64, &'static str>

pub fn qr(&self) -> Result<(Tensor, Tensor), &'static str>

fn get_column(&self, col: usize) -> Self

fn set_column(&mut self, col: usize, tensor: &Self)

pub fn inverse(&self) -> Self

pub fn diag(&self) -> Vec<f64>

pub fn push_value(&mut self, val: f64)

pub fn len(&self) -> usize

pub fn to_vec(&self) -> Vec<f64>

pub fn get_sub_matrix(
    &self,
    left_corner: &[usize],
    shape: &[usize]
) -> Option<Self>

fn copy_recursive(
    &self,
    padded: &mut Tensor,
    index: &mut Vec<usize>,
    ranges: &[Vec<usize>],
    dim: usize
)

pub fn pad(&self, padding: &[(usize, usize)]) -> Option<Self>

fn slice_recursive(
    &self,
    sub_tensor: &mut Tensor,
    index: &mut Vec<usize>,
    ranges: &[(usize, usize)],
    dim: usize
)

pub fn slice(&self, ranges: &[(usize, usize)]) -> Option<Self>


impl Add<Tensor> for Tensor
type Output = Tensor

fn add(self, rhs: Tensor) -> Self::Output

impl Clone for Tensor

fn clone(&self) -> Tensor

fn clone_from(&mut self, : &Self)

impl Debug for Tensor

fn fmt(&self, f: &mut Formatter<'_>) -> Result

impl Default for Tensor

fn default() -> Self

impl Div<Tensor> for Tensor
type Output = Tensor

fn div(self, rhs: Tensor) -> Self::Output

impl Div<f64> for Tensor
type Output = Tensor

fn div(self, rhs: f64) -> Self::Output

impl IntoIterator for Tensor
type IntoIter = TensorIterator
type Item = f64

fn into_iter(self) -> Self::IntoIter

impl Mul<Tensor> for Tensor
type Output = Tensor

fn mul(self, rhs: Tensor) -> Self::Output

impl Mul<Tensor> for f64
type Output = Tensor

fn mul(self, rhs: Tensor) -> Self::Output

impl Mul<f64> for Tensor
type Output = Tensor

fn mul(self, rhs: f64) -> Self::Output

impl Sub<Tensor> for Tensor
type Output = Tensor

fn sub(self, rhs: Tensor) -> Self::Output

impl Sub<f64> for Tensor
type Output = Tensor

fn sub(self, rhs: f64) -> Self::Output
