//! Convert index dialect to LLVM dialect.

use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::Context,
    derive::{op_interface_impl, type_interface_impl},
    irbuild::{inserter::Inserter, match_rewrite::MatchRewriter, rewriter::Rewriter},
    op::Op,
    result::Result,
};
use pliron_llvm::{ToLLVMDialect, ToLLVMType, ToLLVMTypeFn, ops::ConstantOp};

use crate::index::{ops::IndexConstantOp, types::IndexType};

#[type_interface_impl]
impl ToLLVMType for IndexType {
    fn converter(&self) -> ToLLVMTypeFn {
        |_ty, ctx| {
            // Convert IndexType to i64 in LLVM dialect.
            let int_ty = IntegerType::get(ctx, 64, Signedness::Signless);
            Ok(int_ty.into())
        }
    }
}

#[op_interface_impl]
impl ToLLVMDialect for IndexConstantOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let constant_index = self
            .get_attr_constant_index(ctx)
            .expect("Missing value attribute")
            .clone();
        let new_constant_op = ConstantOp::new(ctx, constant_index.into());
        rewriter.insert_op(ctx, new_constant_op);
        rewriter.replace_operation(ctx, self.get_operation(), new_constant_op.get_operation());
        Ok(())
    }
}
