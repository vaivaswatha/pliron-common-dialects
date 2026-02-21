//! Index operations

use pliron::{
    builtin::op_interfaces::{
        AllResultsOfType, NOpdsInterface, NResultsInterface, OneResultInterface,
    },
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
};

use crate::index::{attributes::ConstantIndexAttr, types::IndexType};

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
    attributes = (constant_index: ConstantIndexAttr),
    verifier = "succ",
)]
pub struct IndexConstantOp;

impl IndexConstantOp {
    /// Create a new IndexConstantOp with the given constant index value.
    pub fn new(ctx: &mut Context, constant_index: usize) -> Self {
        let constant_index_attr = ConstantIndexAttr { constant_index };

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![IndexType::get(ctx).into()],
            vec![],
            vec![],
            0,
        );

        let op = Self { op };
        op.set_attr_constant_index(ctx, constant_index_attr);
        op
    }
}
