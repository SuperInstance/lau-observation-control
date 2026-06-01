//! Computability functor ⌊−⌋: 𝒞 → FinSet
//!
//! Maps abstract objects to finite bit representations.
//! This is the "grounding" that makes the categorical framework computable.

use serde::{Deserialize, Serialize};
use nalgebra::DVector;

/// A finite set representation (bit vector).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinSet {
    /// Bits of the representation.
    pub bits: Vec<bool>,
    /// Dimension of the original object.
    pub source_dim: usize,
}

impl FinSet {
    pub fn new(bits: Vec<bool>, source_dim: usize) -> Self {
        Self { bits, source_dim }
    }

    /// Number of bits.
    pub fn len_bits(&self) -> usize {
        self.bits.len()
    }

    /// Convert to integer (if small enough).
    pub fn to_int(&self) -> u64 {
        let mut val = 0u64;
        for (i, &b) in self.bits.iter().enumerate() {
            if b {
                val |= 1u64 << i;
            }
        }
        val
    }

    /// Cardinality of the finite set.
    pub fn cardinality(&self) -> u64 {
        1u64 << self.bits.len()
    }
}

/// Configuration for the computability functor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputabilityConfig {
    /// Bits per dimension.
    pub bits_per_dim: usize,
    /// Scale factor (maps real values to integer range).
    pub scale: f64,
    /// Offset.
    pub offset: f64,
}

impl Default for ComputabilityConfig {
    fn default() -> Self {
        Self {
            bits_per_dim: 8,
            scale: 100.0,
            offset: 0.0,
        }
    }
}

/// The computability functor ⌊−⌋: 𝒞 → FinSet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputabilityFunctor {
    pub config: ComputabilityConfig,
}

impl Default for ComputabilityFunctor {
    fn default() -> Self {
        Self::new(ComputabilityConfig::default())
    }
}

impl ComputabilityFunctor {
    pub fn new(config: ComputabilityConfig) -> Self {
        Self { config }
    }

    /// Map a real value to a finite bit representation.
    pub fn encode_value(&self, value: f64) -> Vec<bool> {
        let scaled = ((value - self.config.offset) * self.config.scale) as i64;
        let max_val = (1i64 << self.config.bits_per_dim) - 1;
        let clamped = scaled.clamp(0, max_val) as u64;
        (0..self.config.bits_per_dim)
            .map(|i| (clamped >> i) & 1 == 1)
            .collect()
    }

    /// Decode a bit representation back to a real value.
    pub fn decode_value(&self, bits: &[bool]) -> f64 {
        let mut val = 0u64;
        for (i, &b) in bits.iter().enumerate() {
            if b {
                val |= 1u64 << i;
            }
        }
        (val as f64) / self.config.scale + self.config.offset
    }

    /// Map an object (DVector) to FinSet.
    pub fn map_object(&self, v: &DVector<f64>) -> FinSet {
        let mut bits = Vec::with_capacity(v.len() * self.config.bits_per_dim);
        for i in 0..v.len() {
            bits.extend(self.encode_value(v[i]));
        }
        FinSet::new(bits, v.len())
    }

    /// Decode FinSet back to DVector.
    pub fn decode_object(&self, finset: &FinSet) -> DVector<f64> {
        let n = finset.source_dim;
        let mut v = DVector::zeros(n);
        for i in 0..n {
            let start = i * self.config.bits_per_dim;
            let end = start + self.config.bits_per_dim;
            if end <= finset.bits.len() {
                v[i] = self.decode_value(&finset.bits[start..end]);
            }
        }
        v
    }

    /// Quantization error for a single value.
    pub fn quantization_error(&self, value: f64) -> f64 {
        let bits = self.encode_value(value);
        let decoded = self.decode_value(&bits);
        (value - decoded).abs()
    }

    /// Maximum representable value.
    pub fn max_value(&self) -> f64 {
        ((1u64 << self.config.bits_per_dim) - 1) as f64 / self.config.scale + self.config.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_encode_decode_roundtrip() {
        let cf = ComputabilityFunctor::default();
        let bits = cf.encode_value(1.5);
        let decoded = cf.decode_value(&bits);
        assert!((decoded - 1.5).abs() < 0.02);
    }

    #[test]
    fn test_map_object_roundtrip() {
        let cf = ComputabilityFunctor::new(ComputabilityConfig {
            bits_per_dim: 8,
            scale: 10.0,
            offset: 0.0,
        });
        let v = dvector![1.0, 2.0, 3.0];
        let finset = cf.map_object(&v);
        assert_eq!(finset.source_dim, 3);
        let decoded = cf.decode_object(&finset);
        for i in 0..3 {
            assert!((decoded[i] - v[i]).abs() < 0.2, "decoded[{}] = {}, expected {}", i, decoded[i], v[i]);
        }
    }

    #[test]
    fn test_finset_cardinality() {
        let fs = FinSet::new(vec![false; 8], 1);
        assert_eq!(fs.cardinality(), 256);
    }

    #[test]
    fn test_finset_to_int() {
        let fs = FinSet::new(vec![true, false, true], 1);
        assert_eq!(fs.to_int(), 5); // bit 0 = 1, bit 2 = 4
    }

    #[test]
    fn test_quantization_error() {
        let cf = ComputabilityFunctor::default();
        let err = cf.quantization_error(1.5);
        assert!(err < 0.02, "Quantization error should be small");
    }

    #[test]
    fn test_custom_config() {
        let cf = ComputabilityFunctor::new(ComputabilityConfig {
            bits_per_dim: 16,
            scale: 1000.0,
            offset: -10.0,
        });
        let bits = cf.encode_value(5.0);
        let decoded = cf.decode_value(&bits);
        assert!((decoded - 5.0).abs() < 0.002);
    }

    #[test]
    fn test_clamping() {
        let cf = ComputabilityFunctor::default();
        // Negative value should clamp to 0
        let bits = cf.encode_value(-100.0);
        let decoded = cf.decode_value(&bits);
        assert!(decoded >= 0.0);
    }
}
