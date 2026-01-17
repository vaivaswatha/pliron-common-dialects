//! Index type

use pliron::{
    derive::{def_type, derive_type_get, format_type},
    impl_verify_succ,
};

/// Index type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[def_type("index.index")]
#[derive_type_get]
#[format_type]
pub struct IndexType;
impl_verify_succ!(IndexType);
