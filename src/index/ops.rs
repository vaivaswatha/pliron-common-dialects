//! Index operations

use pliron::{
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{AllResultsOfType, NOpdsInterface, NResultsInterface, OneResultInterface},
    },
    derive::{def_op, derive_attr_get_set, derive_op_interface_impl, format_op},
    impl_verify_succ,
    op::Op,
};

use crate::index::types::IndexType;

/// Constant operation for IndexType.
#[def_op("index.constant")]
#[derive_op_interface_impl(NOpdsInterface<0>, OneResultInterface, NResultsInterface<1>, AllResultsOfType<IndexType>)]
#[format_op("`<` $constant_index `>` ` : ` type($0)")]
#[derive_attr_get_set(constant_index: IntegerAttr)]
pub struct IndexConstantOp;
impl_verify_succ!(IndexConstantOp);
