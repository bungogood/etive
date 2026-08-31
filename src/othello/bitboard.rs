use std::iter::FusedIterator;

use super::Square;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct BitBoard(pub u64);

impl BitBoard {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u64::MAX);

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn has(self, square: Square) -> bool {
        self.0 & square.bitboard().0 != 0
    }

    #[inline(always)]
    pub const fn iter(self) -> BitBoardIter {
        BitBoardIter(self.0)
    }
}

impl IntoIterator for BitBoard {
    type Item = Square;
    type IntoIter = BitBoardIter;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone)]
pub struct BitBoardIter(u64);

impl Iterator for BitBoardIter {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let index = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(Square::from_index_unchecked(index))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.count_ones() as usize;
        (len, Some(len))
    }
}

impl ExactSizeIterator for BitBoardIter {}
impl FusedIterator for BitBoardIter {}
