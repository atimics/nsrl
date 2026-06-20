use crate::FixedScale;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorView<'a, T> {
    pub shape: &'a [usize],
    pub data: &'a [T],
    pub scale: FixedScale,
}

impl<'a, T> TensorView<'a, T> {
    pub fn new(shape: &'a [usize], data: &'a [T], scale: FixedScale) -> Option<Self> {
        let expected_len = shape
            .iter()
            .try_fold(1_usize, |acc, &dim| acc.checked_mul(dim))?;

        if expected_len != data.len() {
            return None;
        }

        Some(Self { shape, data, scale })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RESIDUAL_Q15_SCALE;

    #[test]
    fn tensor_view_validates_shape_product() {
        assert!(TensorView::new(&[2, 2], &[1_i16, 2, 3, 4], RESIDUAL_Q15_SCALE).is_some());
        assert!(TensorView::new(&[2, 3], &[1_i16, 2, 3, 4], RESIDUAL_Q15_SCALE).is_none());
    }

    #[test]
    fn tensor_view_rejects_shape_overflow() {
        assert!(TensorView::<i16>::new(&[usize::MAX, 2], &[], RESIDUAL_Q15_SCALE).is_none());
    }
}
