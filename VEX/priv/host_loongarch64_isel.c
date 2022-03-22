
/*---------------------------------------------------------------*/
/*--- begin                           host_loongarch64_isel.c ---*/
/*---------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2021-2022 Loongson Technology Corporation Limited

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details->

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.

   The GNU General Public License is contained in the file COPYING.
*/

#include "libvex_basictypes.h"
#include "libvex_ir.h"
#include "libvex.h"

#include "main_util.h"
#include "main_globals.h"
#include "host_generic_regs.h"
#include "host_loongarch64_defs.h"
#include "libvex_guest_offsets.h"  // For OFFSET_loongarch64_COND


/*---------------------------------------------------------*/
/*--- ISelEnv                                           ---*/
/*---------------------------------------------------------*/

/* This carries around:

   - A mapping from IRTemp to IRType, giving the type of any IRTemp we
     might encounter.  This is computed before insn selection starts,
     and does not change.

   - A mapping from IRTemp to HReg.  This tells the insn selector
     which virtual register is associated with each IRTemp temporary.
     This is computed before insn selection starts, and does not
     change.  We expect this mapping to map precisely the same set of
     IRTemps as the type mapping does.

     |vregmap|   holds the primary register for the IRTemp.
     |vregmapHI| is only used for 128-bit integer-typed
                 IRTemps.  It holds the identity of a second
                 64-bit virtual HReg, which holds the high half
                 of the value.

   - The code array, that is, the insns selected so far.

   - A counter, for generating new virtual registers.

   - The host hardware capabilities word.  This is set at the start
     and does not change.

   - A Bool for indicating whether we may generate chain-me
     instructions for control flow transfers, or whether we must use
     XAssisted.

   - The maximum guest address of any guest insn in this block.
     Actually, the address of the highest-addressed byte from any insn
     in this block.  Is set at the start and does not change.  This is
     used for detecting jumps which are definitely forward-edges from
     this block, and therefore can be made (chained) to the fast entry
     point of the destination, thereby avoiding the destination's
     event check.

    - An IRExpr*, which may be NULL, holding the IR expression (an
      IRRoundingMode-encoded value) to which the FPU's rounding mode
      was most recently set.  Setting to NULL is always safe.  Used to
      avoid redundant settings of the FPU's rounding mode, as
      described in set_FPCR_rounding_mode below.

   Note, this is all (well, mostly) host-independent.
*/

typedef
   struct {
      /* Constant -- are set at the start and do not change. */
      IRTypeEnv*   type_env;

      HReg*        vregmap;
      HReg*        vregmapHI;
      Int          n_vregmap;

      UInt         hwcaps;

      Bool         chainingAllowed;
      Addr64       max_ga;

      /* These are modified as we go along. */
      HInstrArray* code;
      Int          vreg_ctr;
   }
   ISelEnv;


static HReg lookupIRTemp ( ISelEnv* env, IRTemp tmp )
{
   vassert(tmp >= 0);
   vassert(tmp < env->n_vregmap);
   return env->vregmap[tmp];
}

static void lookupIRTempPair ( HReg* vrHI, HReg* vrLO,
                               ISelEnv* env, IRTemp tmp )
{
   vassert(tmp < env->n_vregmap);
   vassert(!hregIsInvalid(env->vregmapHI[tmp]));
   *vrLO = env->vregmap[tmp];
   *vrHI = env->vregmapHI[tmp];
}

static void addInstr ( ISelEnv* env, LOONGARCH64Instr* instr )
{
   addHInstr(env->code, instr);
   if (vex_traceflags & VEX_TRACE_VCODE) {
      ppLOONGARCH64Instr(instr, True);
      vex_printf("\n");
   }
}

static HReg newVRegI ( ISelEnv* env )
{
   HReg reg = mkHReg(True/*virtual reg*/, HRcInt64, 0, env->vreg_ctr);
   env->vreg_ctr++;
   return reg;
}

static HReg newVRegF ( ISelEnv* env )
{
   HReg reg = mkHReg(True/*virtual reg*/, HRcFlt64, 0, env->vreg_ctr);
   env->vreg_ctr++;
   return reg;
}


/*---------------------------------------------------------*/
/*--- ISEL: Forward declarations                        ---*/
/*---------------------------------------------------------*/

/* These are organised as iselXXX and iselXXX_wrk pairs.  The
   iselXXX_wrk do the real work, but are not to be called directly.
   For each XXX, iselXXX calls its iselXXX_wrk counterpart, then
   checks that all returned registers are virtual.  You should not
   call the _wrk version directly.
*/

static LOONGARCH64AMode*   iselIntExpr_AMode_wrk ( ISelEnv* env,
                                                   IRExpr* e, IRType dty );
static LOONGARCH64AMode*   iselIntExpr_AMode     ( ISelEnv* env,
                                                   IRExpr* e, IRType dty );

static LOONGARCH64RI*      iselIntExpr_RI_wrk    ( ISelEnv* env, IRExpr* e,
                                                   UChar size, Bool isSigned );
static LOONGARCH64RI*      iselIntExpr_RI        ( ISelEnv* env, IRExpr* e,
                                                   UChar size, Bool isSigned );

static HReg                iselIntExpr_R_wrk     ( ISelEnv* env, IRExpr* e );
static HReg                iselIntExpr_R         ( ISelEnv* env, IRExpr* e );

static void                iselCondCode_wrk      ( ISelEnv* env, IRExpr* e,
                                                   LOONGARCH64CondCode* cc,
                                                   HReg* dst );
static LOONGARCH64CondCode iselCondCode_C        ( ISelEnv* env, IRExpr* e );
static HReg                iselCondCode_R        ( ISelEnv* env, IRExpr* e );

static void                iselInt128Expr_wrk    ( HReg* hi, HReg* lo,
                                                   ISelEnv* env, IRExpr* e );
static void                iselInt128Expr        ( HReg* hi, HReg* lo,
                                                   ISelEnv* env, IRExpr* e );

static HReg                iselFltExpr_wrk        ( ISelEnv* env, IRExpr* e );
static HReg                iselFltExpr            ( ISelEnv* env, IRExpr* e );


/*---------------------------------------------------------*/
/*--- ISEL: Misc helpers                                ---*/
/*---------------------------------------------------------*/

/* Generate move insn */
static LOONGARCH64Instr* LOONGARCH64Instr_Move ( HReg to, HReg from )
{
   LOONGARCH64RI *ri = LOONGARCH64RI_R(hregZERO());
   return LOONGARCH64Instr_Binary(LAbin_OR, ri, from, to);
}

/* Generate LOONGARCH64AMode from HReg and UInt */
static LOONGARCH64AMode* mkLOONGARCH64AMode_RI ( HReg reg, UInt imm )
{
   vassert(imm < (1 << 12));
   return LOONGARCH64AMode_RI(reg, (UShort)imm);
}

/* Store result to guest_COND */
static void store_guest_COND ( ISelEnv* env, HReg src )
{
   UInt offs = OFFSET_loongarch64_COND;
   LOONGARCH64AMode* am = mkLOONGARCH64AMode_RI(hregGSP(), offs);
   addInstr(env, LOONGARCH64Instr_Store(LAstore_ST_W, am, src));
}


/*---------------------------------------------------------*/
/*--- ISEL: Integer expressions (64/32/16/8 bit)        ---*/
/*---------------------------------------------------------*/

/* Select insns for an integer-typed expression, and add them to the
   code list.  Return a reg holding the result.  This reg will be a
   virtual register.  THE RETURNED REG MUST NOT BE MODIFIED.  If you
   want to modify it, ask for a new vreg, copy it in there, and modify
   the copy.  The register allocator will do its best to map both
   vregs to the same real register, so the copies will often disappear
   later in the game.

   This should handle expressions of 64, 32, 16 and 8-bit type.
   All results are returned in a (mode64 ? 64bit : 32bit) register.
   For 16- and 8-bit expressions, the upper (32/48/56 : 16/24) bits
   are arbitrary, so you should mask or sign extend partial values
   if necessary.
*/

/* --------------------- AMode --------------------- */

static LOONGARCH64AMode* iselIntExpr_AMode ( ISelEnv* env,
                                             IRExpr* e, IRType dty )
{
   LOONGARCH64AMode* am = iselIntExpr_AMode_wrk(env, e, dty);

   /* sanity checks ... */
   switch (am->tag) {
      case LAam_RI:
         vassert(am->LAam.RI.index < (1 << 11)); /* The sign bit (bit 12) must be 0. */
         vassert(hregClass(am->LAam.RI.base) == HRcInt64);
         vassert(hregIsVirtual(am->LAam.RI.base));
         break;
      case LAam_RR:
         vassert(hregClass(am->LAam.RR.base) == HRcInt64);
         vassert(hregIsVirtual(am->LAam.RR.base));
         vassert(hregClass(am->LAam.RR.index) == HRcInt64);
         vassert(hregIsVirtual(am->LAam.RR.index));
         break;
      default:
         vpanic("iselIntExpr_AMode: unknown LOONGARCH64 AMode tag");
         break;
   }

   return am;
}

/* DO NOT CALL THIS DIRECTLY ! */
static LOONGARCH64AMode* iselIntExpr_AMode_wrk ( ISelEnv* env,
                                                 IRExpr* e, IRType dty )
{
   IRType ty = typeOfIRExpr(env->type_env, e);
   vassert(e);
   vassert(ty == Ity_I64);

   /* Add64(expr, i), where i <= 0x7ff */
   if (e->tag == Iex_Binop && e->Iex.Binop.op == Iop_Add64
       && e->Iex.Binop.arg2->tag == Iex_Const
       && e->Iex.Binop.arg2->Iex.Const.con->tag == Ico_U64
       && e->Iex.Binop.arg2->Iex.Const.con->Ico.U64 <= 0x7ff) {
      return LOONGARCH64AMode_RI(iselIntExpr_R(env, e->Iex.Binop.arg1),
                                 (UShort)e->Iex.Binop.arg2->Iex.Const.con->Ico.U64);
   }

   /* Add64(expr, expr) */
   if (e->tag == Iex_Binop && e->Iex.Binop.op == Iop_Add64) {
      HReg base = iselIntExpr_R(env, e->Iex.Binop.arg1);
      HReg index = iselIntExpr_R(env, e->Iex.Binop.arg2);
      return LOONGARCH64AMode_RR(base, index);
   }

   /* Doesn't match anything in particular.  Generate it into
      a register and use that. */
   return LOONGARCH64AMode_RI(iselIntExpr_R(env, e), 0);
}

/* --------------------- RI --------------------- */

static LOONGARCH64RI* iselIntExpr_RI ( ISelEnv* env, IRExpr* e,
                                       UChar size, Bool isSigned )
{
   LOONGARCH64RI* ri = iselIntExpr_RI_wrk(env, e, size, isSigned);

   /* sanity checks ... */
   switch (ri->tag) {
      case LAri_Imm:
         vassert(ri->LAri.I.size == 5 || ri->LAri.I.size == 6
                 || ri->LAri.I.size == 12);
         if (ri->LAri.I.size == 5) {
            vassert(ri->LAri.I.isSigned == False);
            vassert(ri->LAri.I.imm < (1 << 5));
         } else if (ri->LAri.I.size == 6) {
            vassert(ri->LAri.I.isSigned == False);
            vassert(ri->LAri.I.imm < (1 << 6));
         } else {
            vassert(ri->LAri.I.imm < (1 << 12));
         }
         break;
      case LAri_Reg:
         vassert(hregClass(ri->LAri.R.reg) == HRcInt64);
         vassert(hregIsVirtual(ri->LAri.R.reg));
         break;
      default:
         vpanic("iselIntExpr_RI: unknown LOONGARCH64 RI tag");
         break;
   }

   return ri;
}

/* DO NOT CALL THIS DIRECTLY ! */
static LOONGARCH64RI* iselIntExpr_RI_wrk ( ISelEnv* env, IRExpr* e,
                                           UChar size, Bool isSigned )
{
   IRType ty = typeOfIRExpr(env->type_env, e);
   vassert(e);
   vassert(ty == Ity_I8 || ty == Ity_I16 || ty == Ity_I32 || ty == Ity_I64);

   LOONGARCH64RI *ri = NULL;

   /* special case: immediate */
   if (e->tag == Iex_Const) {
      switch (e->Iex.Const.con->tag) {
         case Ico_U32:
            if (!isSigned && e->Iex.Const.con->Ico.U32 < (1 << size)) {
               UShort imm = e->Iex.Const.con->Ico.U32;
               ri = LOONGARCH64RI_I(imm, size, isSigned);
            }
            break;
         case Ico_U64:
            if (!isSigned && e->Iex.Const.con->Ico.U64 < (1 << size)) {
               UShort imm = e->Iex.Const.con->Ico.U64;
               ri = LOONGARCH64RI_I(imm, size, isSigned);
            }
            break;
         default:
            break;
      }
      /* else fail, fall through to default case */
   }

   if (ri == NULL) {
      /* default case: calculate into a register and return that */
      HReg reg = iselIntExpr_R(env, e);
      ri = LOONGARCH64RI_R(reg);
   }

   return ri;
}

/* --------------------- Reg --------------------- */

static HReg iselIntExpr_R ( ISelEnv* env, IRExpr* e )
{
   HReg r = iselIntExpr_R_wrk(env, e);

   /* sanity checks ... */
   vassert(hregClass(r) == HRcInt64);
   vassert(hregIsVirtual(r));

   return r;
}

/* DO NOT CALL THIS DIRECTLY ! */
static HReg iselIntExpr_R_wrk ( ISelEnv* env, IRExpr* e )
{
   HReg r;
   return r;
}

/* ------------------- CondCode ------------------- */

/* Generate code to evaluated a bit-typed expression, returning the
   condition code which would correspond when the expression would
   notionally have returned 1. */

static LOONGARCH64CondCode iselCondCode_C ( ISelEnv* env, IRExpr* e )
{
   LOONGARCH64CondCode cc;
   HReg r = newVRegI(env);
   iselCondCode_wrk(env, e, &cc, &r);

   /* sanity checks ... */
   vassert(cc >= LAcc_EQ && cc <= LAcc_AL);

   return cc;
}

static HReg iselCondCode_R ( ISelEnv* env, IRExpr* e )
{
   LOONGARCH64CondCode cc;
   HReg r = newVRegI(env);
   iselCondCode_wrk(env, e, &cc, &r);

   /* sanity checks ... */
   vassert(hregClass(r) == HRcInt64);
   vassert(hregIsVirtual(r));

   return r;
}

/* DO NOT CALL THIS DIRECTLY ! */
static void iselCondCode_wrk ( ISelEnv* env, IRExpr* e,
                               LOONGARCH64CondCode* cc, HReg* dst )
{
}


/*---------------------------------------------------------*/
/*--- ISEL: Integer expressions (128 bit)               ---*/
/*---------------------------------------------------------*/

/* Compute a 128-bit value into a register pair, which is returned as
   the first two parameters.  As with iselIntExpr_R, these may be
   either real or virtual regs; in any case they must not be changed
   by subsequent code emitted by the caller.  */

static void iselInt128Expr(HReg* hi, HReg* lo, ISelEnv* env, IRExpr* e)
{
   iselInt128Expr_wrk(hi, lo, env, e);

   /* sanity checks ... */
   vassert(hregClass(*hi) == HRcInt64);
   vassert(hregIsVirtual(*hi));
   vassert(hregClass(*lo) == HRcInt64);
   vassert(hregIsVirtual(*lo));
}

/* DO NOT CALL THIS DIRECTLY ! */
static void iselInt128Expr_wrk(HReg* hi, HReg* lo, ISelEnv* env, IRExpr* e)
{
}


/*---------------------------------------------------------*/
/*--- ISEL: Floating point expressions (64/32 bit)      ---*/
/*---------------------------------------------------------*/

/* Compute a floating point value into a register, the identity of
   which is returned.  As with iselIntExpr_R, the reg may be either
   real or virtual; in any case it must not be changed by subsequent
   code emitted by the caller.  */

static HReg iselFltExpr ( ISelEnv* env, IRExpr* e )
{
   HReg r = iselFltExpr_wrk(env, e);

   /* sanity checks ... */
   vassert(hregClass(r) == HRcFlt64);
   vassert(hregIsVirtual(r));

   return r;
}

/* DO NOT CALL THIS DIRECTLY */
static HReg iselFltExpr_wrk ( ISelEnv* env, IRExpr* e )
{
   HReg r;
   return r;
}


/*---------------------------------------------------------*/
/*--- ISEL: Statements                                  ---*/
/*---------------------------------------------------------*/

static void iselStmtStore ( ISelEnv* env, IRStmt* stmt )
{
   IRType tya  = typeOfIRExpr(env->type_env, stmt->Ist.Store.addr);
   IRType tyd  = typeOfIRExpr(env->type_env, stmt->Ist.Store.data);

   if (tya != Ity_I64 || stmt->Ist.Store.end != Iend_LE)
      vpanic("iselStmt(loongarch64): Ist_Store");

   Bool                 fp = False;
   LOONGARCH64AMode*    am = iselIntExpr_AMode(env, stmt->Ist.Store.addr, tyd);
   LOONGARCH64StoreOp   op;
   LOONGARCH64FpStoreOp fop;
   switch (tyd) {
      case Ity_I8:
         op = (am->tag == LAam_RI) ? LAstore_ST_B : LAstore_STX_B;
         break;
      case Ity_I16:
         op = (am->tag == LAam_RI) ? LAstore_ST_H : LAstore_STX_H;
         break;
      case Ity_I32:
         op = (am->tag == LAam_RI) ? LAstore_ST_W : LAstore_STX_W;
         break;
      case Ity_I64:
         op = (am->tag == LAam_RI) ? LAstore_ST_D : LAstore_STX_D;
         break;
      case Ity_F32:
         fop = (am->tag == LAam_RI) ? LAfpstore_FST_S : LAfpstore_FSTX_S;
         fp = True;
         break;
      case Ity_F64:
         fop = (am->tag == LAam_RI) ? LAfpstore_FST_D : LAfpstore_FSTX_D;
         fp = True;
         break;
      default:
         vpanic("iselStmt(loongarch64): Ist_Store");
         break;
   }

   if (fp) {
      HReg src = iselFltExpr(env, stmt->Ist.Store.data);
      addInstr(env, LOONGARCH64Instr_FpStore(fop, am, src));
   } else {
      HReg src = iselIntExpr_R(env, stmt->Ist.Store.data);
      addInstr(env, LOONGARCH64Instr_Store(op, am, src));
   }
}

static void iselStmtPut ( ISelEnv* env, IRStmt* stmt )
{
   IRType ty = typeOfIRExpr(env->type_env, stmt->Ist.Put.data);

   Bool                 fp = False;
   LOONGARCH64AMode*    am = mkLOONGARCH64AMode_RI(hregGSP(), stmt->Ist.Put.offset);
   LOONGARCH64StoreOp   op;
   LOONGARCH64FpStoreOp fop;
   switch (ty) {
      case Ity_I8:
         op = LAstore_ST_B;
         break;
      case Ity_I16:
         op = LAstore_ST_H;
         break;
      case Ity_I32:
         op = LAstore_ST_W;
         break;
      case Ity_I64:
         op = LAstore_ST_D;
         break;
      case Ity_F32:
         fop = LAfpstore_FST_S;
         fp = True;
         break;
      case Ity_F64:
         fop = LAfpstore_FST_D;
         fp = True;
         break;
      default:
         vpanic("iselStmt(loongarch64): Ist_Put");
         break;
   }

   if (fp) {
      HReg src = iselFltExpr(env, stmt->Ist.Put.data);
      addInstr(env, LOONGARCH64Instr_FpStore(fop, am, src));
   } else {
      HReg src = iselIntExpr_R(env, stmt->Ist.Put.data);
      addInstr(env, LOONGARCH64Instr_Store(op, am, src));
   }
}

static void iselStmtTmp ( ISelEnv* env, IRStmt* stmt )
{
   IRTemp tmp = stmt->Ist.WrTmp.tmp;
   IRType ty  = typeOfIRTemp(env->type_env, tmp);

   if (ty == Ity_I8 || ty == Ity_I16 || ty == Ity_I32 || ty == Ity_I64) {
      HReg dst = lookupIRTemp(env, tmp);
      HReg src = iselIntExpr_R(env, stmt->Ist.WrTmp.data);
      addInstr(env, LOONGARCH64Instr_Move(dst, src));
   } else if (ty == Ity_I1) {
      HReg dst = lookupIRTemp(env, tmp);
      HReg src = iselCondCode_R(env, stmt->Ist.WrTmp.data);
      addInstr(env, LOONGARCH64Instr_Move(dst, src));
   } else if (ty == Ity_F32) {
      HReg dst = lookupIRTemp(env, tmp);
      HReg src = iselFltExpr(env, stmt->Ist.WrTmp.data);
      addInstr(env, LOONGARCH64Instr_FpMove(LAfpmove_FMOV_S, src, dst));
   } else if (ty == Ity_F64) {
      HReg dst = lookupIRTemp(env, tmp);
      HReg src = iselFltExpr(env, stmt->Ist.WrTmp.data);
      addInstr(env, LOONGARCH64Instr_FpMove(LAfpmove_FMOV_D, src, dst));
   } else {
      vpanic("iselStmt(loongarch64): Ist_WrTmp");
   }
}

static void iselStmtDirty ( ISelEnv* env, IRStmt* stmt )
{
   IRDirty* d = stmt->Ist.Dirty.details;

   /* Figure out the return type, if any. */
   IRType retty = Ity_INVALID;
   if (d->tmp != IRTemp_INVALID)
      retty = typeOfIRTemp(env->type_env, d->tmp);

   Bool retty_ok = False;
   switch (retty) {
      case Ity_INVALID: /* function doesn't return anything */
      case Ity_I8: case Ity_I16: case Ity_I32: case Ity_I64:
         retty_ok = True;
         break;
      default:
         break;
   }
   if (!retty_ok)
      vpanic("iselStmt(loongarch64): Ist_Dirty");

   /* Marshal args, do the call, and set the return value to 0x555..555
      if this is a conditional call that returns a value and the
      call is skipped. */
   UInt   addToSp = 0;
   RetLoc rloc    = mk_RetLoc_INVALID();
   doHelperCall(&addToSp, &rloc, env, d->guard, d->cee, retty, d->args);
   vassert(is_sane_RetLoc(rloc));

   /* Now figure out what to do with the returned value, if any. */
   switch (retty) {
      case Ity_INVALID: {
         /* No return value.  Nothing to do. */
         vassert(d->tmp == IRTemp_INVALID);
         vassert(rloc.pri == RLPri_None);
         vassert(addToSp == 0);
         break;
      }
      case Ity_I8: case Ity_I16: case Ity_I32: case Ity_I64: {
         vassert(rloc.pri == RLPri_Int);
         vassert(addToSp == 0);
         /* The returned value is in $a0.  Park it in the register
            associated with tmp. */
         HReg dst = lookupIRTemp(env, d->tmp);
         addInstr(env, LOONGARCH64Instr_Move(dst, hregLOONGARCH64_R4()));
         break;
      }
      default:
         /*NOTREACHED*/
         vassert(0);
         break;
   }
}

static void iselStmtLLSC ( ISelEnv* env, IRStmt* stmt )
{
   IRTemp res = stmt->Ist.LLSC.result;
   IRType tya = typeOfIRExpr(env->type_env, stmt->Ist.LLSC.addr);

   /* Temporary solution; this need to be rewritten again for LOONGARCH64.
      On LOONGARCH64 you can not read from address that is locked with LL
      before SC. If you read from address that is locked than SC will fall.
    */
   if (stmt->Ist.LLSC.storedata == NULL) {
      /* LL */
      IRType ty = typeOfIRTemp(env->type_env, res);
      LOONGARCH64LLSCOp op;
      switch (ty) {
         case Ity_I32:
            op = LAllsc_LL_W;
            break;
         case Ity_I64:
            op = LAllsc_LL_D;
            break;
         default:
            vpanic("iselStmt(loongarch64): Ist_LLSC");
            break;
      }
      LOONGARCH64AMode* addr = iselIntExpr_AMode(env, stmt->Ist.LLSC.addr, tya);
      HReg               val = lookupIRTemp(env, res);
      addInstr(env, LOONGARCH64Instr_LLSC(op, True, addr, val));
   } else {
      /* SC */
      IRType tyd = typeOfIRExpr(env->type_env, stmt->Ist.LLSC.storedata);
      LOONGARCH64LLSCOp op;
      switch (tyd) {
         case Ity_I32:
            op = LAllsc_SC_W;
            break;
         case Ity_I64:
            op = LAllsc_SC_D;
            break;
         default:
            vpanic("iselStmt(loongarch64): Ist_LLSC");
            break;
      }
      LOONGARCH64AMode* addr = iselIntExpr_AMode(env, stmt->Ist.LLSC.addr, tya);
      HReg               val = iselIntExpr_R(env, stmt->Ist.LLSC.storedata);
      HReg               dst = lookupIRTemp(env, res);
      HReg               tmp = newVRegI(env);
      addInstr(env, LOONGARCH64Instr_Move(tmp, val));
      addInstr(env, LOONGARCH64Instr_LLSC(op, False, addr, tmp));
      addInstr(env, LOONGARCH64Instr_Move(dst, tmp));
   }
}

static void iselStmtCas ( ISelEnv* env, IRStmt* stmt )
{
   IRCAS* cas = stmt->Ist.CAS.details;
   if (cas->oldHi == IRTemp_INVALID && cas->end == Iend_LE) {
      /* "normal" singleton CAS */
      HReg   old = lookupIRTemp(env, cas->oldLo);
      HReg  addr = iselIntExpr_R(env, cas->addr);
      HReg  expd = iselIntExpr_R(env, cas->expdLo);
      HReg  data = iselIntExpr_R(env, cas->dataLo);
      IRType  ty = typeOfIRTemp(env->type_env, cas->oldLo);
      Bool size64;
      switch (ty) {
         case Ity_I32:
            size64 = False;
            break;
         case Ity_I64:
            size64 = True;
            break;
         default:
            vpanic("iselStmt(loongarch64): Ist_CAS");
            break;
      }
      addInstr(env, LOONGARCH64Instr_Cas(old, addr, expd, data, size64));
   } else {
      vpanic("iselStmt(loongarch64): Ist_CAS");
   }
}

static void iselStmtMBE ( ISelEnv* env, IRStmt* stmt )
{
   switch (stmt->Ist.MBE.event) {
      case Imbe_Fence:
      case Imbe_CancelReservation:
         addInstr(env, LOONGARCH64Instr_Bar(LAbar_DBAR, 0));
         break;
      case Imbe_InsnFence:
         addInstr(env, LOONGARCH64Instr_Bar(LAbar_IBAR, 0));
         break;
      default:
         vpanic("iselStmt(loongarch64): Ist_MBE");
         break;
   }
}

static void iselStmtExit ( ISelEnv* env, IRStmt* stmt )
{
   if (stmt->Ist.Exit.dst->tag != Ico_U64)
      vpanic("iselStmt(loongarch64): Ist_Exit: dst is not a 64-bit value");

   LOONGARCH64CondCode cc = iselCondCode_C(env, stmt->Ist.Exit.guard);
   LOONGARCH64AMode*   am = mkLOONGARCH64AMode_RI(hregGSP(), stmt->Ist.Exit.offsIP);

   /* Case: boring transfer to known address */
   if (stmt->Ist.Exit.jk == Ijk_Boring) {
      if (env->chainingAllowed) {
         /* .. almost always true .. */
         /* Skip the event check at the dst if this is a forwards edge. */
         Bool toFastEP = ((Addr64)stmt->Ist.Exit.dst->Ico.U64) > env->max_ga;
         addInstr(env, LOONGARCH64Instr_XDirect(stmt->Ist.Exit.dst->Ico.U64,
                                                am, cc, toFastEP));
      } else {
         /* .. very occasionally .. */
         /* We can't use chaining, so ask for an assisted transfer,
            as that's the only alternative that is allowable. */
         HReg dst = iselIntExpr_R(env, IRExpr_Const(stmt->Ist.Exit.dst));
         addInstr(env, LOONGARCH64Instr_XAssisted(dst, am, cc, Ijk_Boring));
      }
      return;
   }

   /* Case: assisted transfer to arbitrary address */
   switch (stmt->Ist.Exit.jk) {
      /* Keep this list in sync with that for iselNext below */
      case Ijk_ClientReq:
      case Ijk_Yield:
      case Ijk_NoDecode:
      case Ijk_InvalICache:
      case Ijk_NoRedir:
      case Ijk_SigTRAP:
      case Ijk_SigSEGV:
      case Ijk_Sys_syscall: {
         HReg dst = iselIntExpr_R(env, IRExpr_Const(stmt->Ist.Exit.dst));
         addInstr(env, LOONGARCH64Instr_XAssisted(dst, am, cc, stmt->Ist.Exit.jk));
         break;
      }
      default:
         /* Do we ever expect to see any other kind? */
         vpanic("iselStmt(loongarch64): Ist_Exit");
         break;
   }
}

static void iselStmt(ISelEnv* env, IRStmt* stmt)
{
   if (vex_traceflags & VEX_TRACE_VCODE) {
      vex_printf("\n-- ");
      ppIRStmt(stmt);
      vex_printf("\n");
   }

   switch (stmt->tag) {
      /* --------- STORE --------- */
      /* little-endian write to memory */
      case Ist_Store:
         iselStmtStore(env, stmt);
         break;

      /* --------- PUT --------- */
      /* write guest state, fixed offset */
      case Ist_Put:
         iselStmtPut(env, stmt);
         break;

      /* --------- TMP --------- */
      /* assign value to temporary */
      case Ist_WrTmp:
         iselStmtTmp(env, stmt);
         break;

      /* --------- Call to DIRTY helper --------- */
      /* call complex ("dirty") helper function */
      case Ist_Dirty:
         iselStmtDirty(env, stmt);
         break;

      /* --------- Load Linked and Store Conditional --------- */
      case Ist_LLSC:
         iselStmtLLSC(env, stmt);
         break;

      /* --------- CAS --------- */
      case Ist_CAS:
         iselStmtCas(env, stmt);
         break;

      /* --------- MEM FENCE --------- */
      case Ist_MBE:
         iselStmtMBE(env, stmt);
         break;

      /* --------- INSTR MARK --------- */
      /* Doesn't generate any executable code ... */
      case Ist_IMark:
         break;

      /* --------- ABI HINT --------- */
      /* These have no meaning (denotation in the IR) and so we ignore
         them ... if any actually made it this far. */
      case Ist_AbiHint:
         break;

      /* --------- NO-OP --------- */
      case Ist_NoOp:
         break;

      /* --------- EXIT --------- */
      case Ist_Exit:
         iselStmtExit(env, stmt);
         break;

      default:
         ppIRStmt(stmt);
         vpanic("iselStmt(loongarch64)");
         break;
   }
}


/*---------------------------------------------------------*/
/*--- ISEL: Basic block terminators (Nexts)             ---*/
/*---------------------------------------------------------*/

static void iselNext ( ISelEnv* env, IRExpr* next, IRJumpKind jk, Int offsIP )
{
   if (vex_traceflags & VEX_TRACE_VCODE) {
      vex_printf("\n-- PUT(%d) = ", offsIP);
      ppIRExpr(next);
      vex_printf("; exit-");
      ppIRJumpKind(jk);
      vex_printf("\n");
   }

   /* Case: boring transfer to known address */
   if (next->tag == Iex_Const) {
      IRConst* cdst = next->Iex.Const.con;
      vassert(cdst->tag == Ico_U64);
      if (jk == Ijk_Boring) {
         /* Boring transfer to known address */
         LOONGARCH64AMode* am = mkLOONGARCH64AMode_RI(hregGSP(), offsIP);
         if (env->chainingAllowed) {
            /* .. almost always true .. */
            /* Skip the event check at the dst if this is a forwards edge. */
            Bool toFastEP = ((Addr64)cdst->Ico.U64) > env->max_ga;
            addInstr(env, LOONGARCH64Instr_XDirect(cdst->Ico.U64, am, LAcc_AL, toFastEP));
         } else {
            /* .. very occasionally .. */
            /* We can't use chaining, so ask for an assisted transfer,
               as that's the only alternative that is allowable. */
            HReg dst = iselIntExpr_R(env, next);
            addInstr(env, LOONGARCH64Instr_XAssisted(dst, am, LAcc_AL, Ijk_Boring));
         }
         return;
      }
   }

   /* Case: boring transfer to any address */
   switch (jk) {
      case Ijk_Boring: {
         HReg dst = iselIntExpr_R(env, next);
         LOONGARCH64AMode* am = mkLOONGARCH64AMode_RI(hregGSP(), offsIP);
         if (env->chainingAllowed) {
            addInstr(env, LOONGARCH64Instr_XIndir(dst, am, LAcc_AL));
         } else {
            addInstr(env, LOONGARCH64Instr_XAssisted(dst, am, LAcc_AL, Ijk_Boring));
         }
         return;
      }
      default:
         break;
   }

   /* Case: assisted transfer to arbitrary address */
   switch (jk) {
      /* Keep this list in sync with that for Ist_Exit above */
      case Ijk_ClientReq:
      case Ijk_NoDecode:
      case Ijk_InvalICache:
      case Ijk_NoRedir:
      case Ijk_SigTRAP:
      case Ijk_SigSEGV:
      case Ijk_Sys_syscall: {
         HReg dst = iselIntExpr_R(env, next);
         LOONGARCH64AMode* am = mkLOONGARCH64AMode_RI(hregGSP(), offsIP);
         addInstr(env, LOONGARCH64Instr_XAssisted(dst, am, LAcc_AL, jk));
         return;
      }
      default:
         break;
   }

   vex_printf("\n-- PUT(%d) = ", offsIP);
   ppIRExpr(next);
   vex_printf("; exit-");
   ppIRJumpKind(jk);
   vex_printf("\n");
   vassert(0); // are we expecting any other kind?
}


/*---------------------------------------------------------*/
/*--- Insn selector top-level                           ---*/
/*---------------------------------------------------------*/

/* Translate an entire BB to LOONGARCH64 code. */
HInstrArray* iselSB_LOONGARCH64 ( const IRSB* bb,
                                  VexArch arch_host,
                                  const VexArchInfo* archinfo_host,
                                  const VexAbiInfo* vbi,
                                  Int offs_Host_EvC_Counter,
                                  Int offs_Host_EvC_FailAddr,
                                  Bool chainingAllowed,
                                  Bool addProfInc,
                                  Addr max_ga )
{
   Int        i, j;
   HReg       hreg, hregHI;
   ISelEnv*   env;
   UInt       hwcaps_host = archinfo_host->hwcaps;
   LOONGARCH64AMode *amCounter, *amFailAddr;

   /* sanity ... */
   vassert(arch_host == VexArchLOONGARCH64);
   vassert((hwcaps_host & ~(VEX_HWCAPS_LOONGARCH_CPUCFG
                          | VEX_HWCAPS_LOONGARCH_LAM
                          | VEX_HWCAPS_LOONGARCH_UAL
                          | VEX_HWCAPS_LOONGARCH_FP
                          | VEX_HWCAPS_LOONGARCH_LSX
                          | VEX_HWCAPS_LOONGARCH_LASX
                          | VEX_HWCAPS_LOONGARCH_COMPLEX
                          | VEX_HWCAPS_LOONGARCH_CRYPTO
                          | VEX_HWCAPS_LOONGARCH_LVZP
                          | VEX_HWCAPS_LOONGARCH_X86BT
                          | VEX_HWCAPS_LOONGARCH_ARMBT
                          | VEX_HWCAPS_LOONGARCH_MIPSBT
                          | VEX_HWCAPS_LOONGARCH_ISA_32BIT
                          | VEX_HWCAPS_LOONGARCH_ISA_64BIT)) == 0);

   /* Check that the host's endianness is as expected. */
   vassert(archinfo_host->endness == VexEndnessLE);

   /* Make up an initial environment to use. */
   env = LibVEX_Alloc_inline(sizeof(ISelEnv));
   env->vreg_ctr = 0;

   /* Set up output code array. */
   env->code = newHInstrArray();

   /* Copy BB's type env. */
   env->type_env = bb->tyenv;

   /* Make up an IRTemp -> virtual HReg mapping.  This doesn't
      change as we go along. */
   env->n_vregmap = bb->tyenv->types_used;
   env->vregmap   = LibVEX_Alloc_inline(env->n_vregmap * sizeof(HReg));
   env->vregmapHI = LibVEX_Alloc_inline(env->n_vregmap * sizeof(HReg));

   /* and finally ... */
   env->chainingAllowed = chainingAllowed;
   env->hwcaps          = hwcaps_host;
   env->max_ga          = max_ga;

   /* For each IR temporary, allocate a suitably-kinded virtual register. */
   j = 0;
   for (i = 0; i < env->n_vregmap; i++) {
      hregHI = hreg = INVALID_HREG;
      switch (bb->tyenv->types[i]) {
         case Ity_I1:
         case Ity_I8:
         case Ity_I16:
         case Ity_I32:
         case Ity_I64:
            hreg = mkHReg(True, HRcInt64, 0, j++);
            break;
         case Ity_I128:
            hreg   = mkHReg(True, HRcInt64, 0, j++);
            hregHI = mkHReg(True, HRcInt64, 0, j++);
            break;
         case Ity_F16: // we'll use HRcFlt64 regs for F16 too
         case Ity_F32: // we'll use HRcFlt64 regs for F32 too
         case Ity_F64:
            hreg = mkHReg(True, HRcFlt64, 0, j++);
            break;
         default:
            ppIRType(bb->tyenv->types[i]);
            vpanic("iselBB(loongarch64): IRTemp type");
      }
      env->vregmap[i]   = hreg;
      env->vregmapHI[i] = hregHI;
   }
   env->vreg_ctr = j;

   /* The very first instruction must be an event check. */
   amCounter  = mkLOONGARCH64AMode_RI(hregGSP(), offs_Host_EvC_Counter);
   amFailAddr = mkLOONGARCH64AMode_RI(hregGSP(), offs_Host_EvC_FailAddr);
   addInstr(env, LOONGARCH64Instr_EvCheck(amCounter, amFailAddr));

   /* Possibly a block counter increment (for profiling).  At this
      point we don't know the address of the counter, so just pretend
      it is zero.  It will have to be patched later, but before this
      translation is used, by a call to LibVEX_patchProfCtr. */
   if (addProfInc) {
      addInstr(env, LOONGARCH64Instr_ProfInc());
   }

   /* Ok, finally we can iterate over the statements. */
   for (i = 0; i < bb->stmts_used; i++)
      iselStmt(env, bb->stmts[i]);

   iselNext(env, bb->next, bb->jumpkind, bb->offsIP);

   /* record the number of vregs we used. */
   env->code->n_vregs = env->vreg_ctr;
   return env->code;
}


/*---------------------------------------------------------------*/
/*--- end                             host_loongarch64_isel.c ---*/
/*---------------------------------------------------------------*/
