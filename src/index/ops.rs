//! Index operations

use pliron::{
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{AllResultsOfType, NOpdsInterface, NResultsInterface, OneResultInterface},
    },
    derive::pliron_op,
};

use crate::index::types::IndexType;

/// Constant operation for IndexType.
#[pliron_op(
    name = "index.constant",
    interfaces = [
        NOpdsInterface<0>,
        OneResultInterface,
        NResultsInterface<1>,
        AllResultsOfType<IndexType>
    ],
    format = "`<` $constant_index `>` ` : ` type($0)",
    attributes = (constant_index: IntegerAttr),
    verifier = "succ",
)]
pub struct IndexConstantOp;
