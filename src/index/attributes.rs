//! Index dialect attributes

use pliron::derive::pliron_attr;

/// Attribute for index constant values.
#[pliron_attr(name = "index.constant", format = "$constant_index", verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstantIndexAttr {
    /// The constant index value.
    pub constant_index: usize,
}
