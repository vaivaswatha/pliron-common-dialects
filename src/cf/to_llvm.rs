//! Convert control flow dialect to LLVM dialect.

use pliron::{
    builtin::op_interfaces::{OneRegionInterface, OneResultInterface},
    context::{Context, Ptr},
    derive::op_interface_impl,
    input_error,
    irbuild::{
        inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint},
        match_rewrite::{MatchRewrite, MatchRewriter},
        rewriter::Rewriter,
    },
    op::{Op, op_cast, op_impls},
    operation::Operation,
    result::Result,
    r#type::{Typed, type_cast},
};
use pliron_llvm::{
    ToLLVMDialect, ToLLVMType,
    attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr},
    op_interfaces::IntBinArithOpWithOverflowFlag,
    ops::{AddOp, BrOp, CondBrOp, ICmpOp},
};

use crate::cf::ops::ForOp;

/// Implement [MatchRewrite] for control-flow to LLVM conversion.
pub struct CFToLLVM;

impl MatchRewrite for CFToLLVM {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToLLVMDialect>(&*Operation::get_op_dyn(op, ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_llvm_op = op_cast::<dyn ToLLVMDialect>(&*op_dyn)
            .expect("Matched Op must implement ToLLVMDialect");
        to_llvm_op.rewrite(ctx, rewriter)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ForOpConversionErr {
    #[error("Unsupported induction variable type for ForOp conversion")]
    UnsupportedIVType,
}

// Rewrite pattern for `ForOp`:
//     <code before ForOp>
//     llvm.br ^header(%lower_bound, %iter_args_init, ...)
//
//  ^header(%iv, %iter_args, ...):
//     %cmp = llvm.icmp %iv LT %upper_bound
//     llvm.cond_br if %cmp ^for_body_entry(%iv, %iter_args) else ^exit
//
//  ^for_body_entry(%iv, %iter_args, ...):
//     ...
//
//  ^for_body_exit(...):
//     ...
//     %yield_operands, ... = ... # remove `yield` op
//     %iv_next = llvm.add %iv, 1
//     llvm.br ^header(%iv_next, %yield_operands, ...)
//
//  ^exit:
//     <code after ForOp>
#[op_interface_impl]
impl ToLLVMDialect for ForOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let lower_bound = self.get_lower_bound(ctx);
        let upper_bound = self.get_upper_bound(ctx);
        let step = self.get_step(ctx);
        let iter_vars_init = self.get_iter_args_init(ctx);
        let iv = self.get_induction_variable(ctx);
        let for_body_entry = self.get_entry(ctx);

        let self_op = self.get_operation();

        let pre_header = self_op
            .deref(ctx)
            .get_parent_block()
            .expect("ForOp must be inside a block");

        let exit_block =
            rewriter.split_block(ctx, pre_header, OpInsertionPoint::BeforeOperation(self_op));

        let iv_ty = iv.get_type(ctx);
        let to_llvm_ty = type_cast::<dyn ToLLVMType>(&**iv_ty.deref(ctx))
            .ok_or_else(|| input_error!(self.loc(ctx), ForOpConversionErr::UnsupportedIVType))?
            .converter();
        let iv_ty = to_llvm_ty(iv_ty, ctx)?;
        // We change the type of the induction variable to an LLVM integer type.
        iv.set_type(ctx, iv_ty);

        // We don't convert iter_var_types here because they are just passed
        // through the header without any operations on them. They will be converted
        // when they are used in the body block.
        let iter_var_types = iter_vars_init.iter().map(|arg| arg.get_type(ctx));

        // Setup the header
        let header = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AfterBlock(pre_header),
            Some("for_op_header".try_into().unwrap()),
            std::iter::once(iv_ty).chain(iter_var_types).collect(),
        );
        let header_iv = header.deref(ctx).get_argument(0);
        let header_args = header.deref(ctx).arguments().collect::<Vec<_>>();
        let cmp = ICmpOp::new(ctx, ICmpPredicateAttr::ULT, header_iv, upper_bound);
        let cmp_result = cmp.get_result(ctx);
        rewriter.insert_op(ctx, cmp);
        let cond_br = CondBrOp::new(
            ctx,
            cmp_result,
            for_body_entry,
            header_args.clone(),
            exit_block,
            vec![],
        );
        rewriter.insert_op(ctx, cond_br);

        // Pre-header must branch to header with initial induction variable and iter args
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(pre_header));
        let pre_header_br = BrOp::new(
            ctx,
            header,
            std::iter::once(lower_bound)
                .chain(iter_vars_init.iter().cloned())
                .collect(),
        );
        rewriter.insert_op(ctx, pre_header_br);

        // Set the for body exit block to to branch to the header
        // with the next induction variable and iter args from yield.
        let for_body_entry_iv = for_body_entry.deref(ctx).get_argument(0);
        let yield_op = self.get_yield(ctx).get_operation();
        rewriter.set_insertion_point(OpInsertionPoint::AfterOperation(yield_op));
        let iv_next = AddOp::new_with_overflow_flag(
            ctx,
            for_body_entry_iv,
            step,
            IntegerOverflowFlagsAttr {
                nsw: false,
                nuw: false,
            },
        );
        rewriter.append_op(ctx, iv_next);
        let branch_operands: Vec<_> = std::iter::once(iv_next.get_result(ctx))
            .chain(yield_op.deref(ctx).operands())
            .collect();
        let for_body_exit_br = BrOp::new(ctx, header, branch_operands);
        rewriter.append_op(ctx, for_body_exit_br);
        rewriter.erase_operation(ctx, yield_op);

        // Inline the body of the ForOp, to after the header block.
        rewriter.inline_region(
            ctx,
            self.get_region(ctx),
            BlockInsertionPoint::AfterBlock(header),
        );

        // Replace uses of the ForOp with header block arguments
        // (except the induction variable which is not used outside).
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            header_args.into_iter().skip(1).collect(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pliron::{
        builtin::ops::ModuleOp,
        combine::Parser,
        context::Context,
        input_error_noloc,
        irbuild::match_rewrite::collect_rewrite,
        irfmt::parsers::spaced,
        location,
        op::verify_op,
        operation::Operation,
        parsable::{self, state_stream_from_iterator},
        printable::Printable,
        result::ExpectOk,
    };
    use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};

    use crate::cf::to_llvm::CFToLLVM;
    use expect_test::expect;

    #[test]
    fn test_for_op_to_llvm_conversion() {
        let ctx = &mut Context::new();

        let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_for: llvm.func <builtin.fp32 () variadic = false> [] {
                  ^entry():
                    c0 = index.constant <builtin.integer <0: i64>> : index.index;
                    c10 = index.constant <builtin.integer <10: i64>> : index.index;
                    c1 = index.constant <builtin.integer <1: i64>> : index.index;
                    init = llvm.constant <builtin.single 1.0> : builtin.fp32;
                    inc = llvm.constant <builtin.single 3.5> : builtin.fp32;
                    
                    result = cf.for c0 to c10 step c1 (init) {
                        ^entry(iv : index.index, iter_arg : builtin.fp32):
                            next = llvm.fadd <FAST> iter_arg, inc : builtin.fp32;
                            cf.yield next
                    };
                    
                    llvm.return result
                }
            }
            "#;

        let state_stream = state_stream_from_iterator(
            input_ir.chars(),
            parsable::State::new(ctx, location::Source::InMemory),
        );
        let parsed = spaced(Operation::top_level_parser())
            .parse(state_stream)
            .map(|(op, _)| op)
            .map_err(|err| input_error_noloc!(err));

        let parsed_op = parsed.expect_ok(ctx);
        let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
        verify_op(&module_op, ctx).expect_ok(ctx);

        collect_rewrite(ctx, CFToLLVM, parsed_op).expect_ok(ctx);
        verify_op(&module_op, ctx).expect_ok(ctx);

        let print_parsed = format!("{}", module_op.disp(ctx));
        expect![[r#"
                builtin.module @test_module 
                {
                  ^entry_block3v1():
                    llvm.func @test_for: llvm.func <builtin.fp32 () variadic = false>
                      [] 
                    {
                      ^entry_block2v1():
                        op12v1_res0 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
                        op3v3_res0 = llvm.constant <builtin.integer <10: i64>> : builtin.integer i64;
                        op4v3_res0 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                        init_op6v1_res0 = llvm.constant <builtin.single 1> : builtin.fp32  !0;
                        inc_op7v1_res0 = llvm.constant <builtin.single 3.5> : builtin.fp32  !1;
                        llvm.br ^for_op_header_block5v1(op12v1_res0, init_op6v1_res0)

                      ^for_op_header_block5v1(block5v1_arg0: builtin.integer i64, block5v1_arg1: builtin.fp32 ):
                        op5v3_res0 = llvm.icmp block5v1_arg0 <ULT> op3v3_res0 : builtin.integer i1;
                        llvm.cond_br if op5v3_res0 ^entry_block1v1(block5v1_arg0, block5v1_arg1) else ^entry_split_block4v1()

                      ^entry_block1v1(iv_block1v1_arg0: builtin.integer i64, iter_arg_block1v1_arg1: builtin.fp32 ):
                        next_op10v1_res0 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> iter_arg_block1v1_arg1, inc_op7v1_res0 : builtin.fp32  !2;
                        op15v1_res0 = llvm.add iv_block1v1_arg0, op4v3_res0 <{nsw=false,nuw=false}>: builtin.integer i64;
                        llvm.br ^for_op_header_block5v1(op15v1_res0, next_op10v1_res0)

                      ^entry_split_block4v1():
                        llvm.return block5v1_arg1 !3
                    } !4
                }"#]].assert_eq(&print_parsed);

        let llvm_ctx = LLVMContext::default();
        let llvm_ir =
            pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
        llvm_ir
            .verify()
            .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
            .unwrap();

        expect![[r#"
                ; ModuleID = 'test_module'
                source_filename = "test_module"

                define float @test_for() {
                entry_block2v1:
                  br label %for_op_header_block5v1

                for_op_header_block5v1:                           ; preds = %entry_block1v1, %entry_block2v1
                  %block5v1_arg0 = phi i64 [ 0, %entry_block2v1 ], [ %op15v1_res0, %entry_block1v1 ]
                  %block5v1_arg1 = phi float [ 1.000000e+00, %entry_block2v1 ], [ %next_op10v1_res0, %entry_block1v1 ]
                  %op5v3_res0 = icmp ult i64 %block5v1_arg0, 10
                  br i1 %op5v3_res0, label %entry_block1v1, label %entry_split_block4v1

                entry_block1v1:                                   ; preds = %for_op_header_block5v1
                  %iv_block1v1_arg0 = phi i64 [ %block5v1_arg0, %for_op_header_block5v1 ]
                  %iter_arg_block1v1_arg1 = phi float [ %block5v1_arg1, %for_op_header_block5v1 ]
                  %next_op10v1_res0 = fadd fast float %iter_arg_block1v1_arg1, 3.500000e+00
                  %op15v1_res0 = add i64 %iv_block1v1_arg0, 1
                  br label %for_op_header_block5v1

                entry_split_block4v1:                             ; preds = %for_op_header_block5v1
                  ret float %block5v1_arg1
                }
            "#]].assert_eq(&llvm_ir.to_string());

        // Let's try and execute this function
        initialize_native().expect("Failed to initialize native target for LLVM execution");
        let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
        jit.add_module(llvm_ir)
            .expect("Failed to add module to JIT");
        let symbol_addr = jit
            .lookup_symbol("test_for")
            .expect("Failed to lookup symbol");
        assert!(symbol_addr != 0);
        let f = unsafe { std::mem::transmute::<u64, fn() -> f32>(symbol_addr) };
        let result = f();
        assert_eq!(result, 36.0);
    }
}
