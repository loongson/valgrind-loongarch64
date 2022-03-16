
/*---------------------------------------------------------------*/
/*--- begin                       guest_loongarch64_helpers.c ---*/
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
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.

   The GNU General Public License is contained in the file COPYING.
*/

#include "libvex_basictypes.h"
#include "libvex_emnote.h"
#include "libvex_guest_loongarch64.h"
#include "libvex_ir.h"
#include "libvex.h"

#include "main_util.h"
#include "main_globals.h"
#include "guest_generic_bb_to_IR.h"
#include "guest_loongarch64_defs.h"


/* This file contains helper functions for loongarch64 guest code.
   Calls to these functions are generated by the back end. */

IRExpr* guest_loongarch64_spechelper ( const HChar * function_name,
                                       IRExpr ** args,
                                       IRStmt ** precedingStmts,
                                       Int n_precedingStmts )
{
   return NULL;
}

/* VISIBLE TO LIBVEX CLIENT */
void LibVEX_GuestLOONGARCH64_initialise ( /*OUT*/
                                          VexGuestLOONGARCH64State* vex_state )
{
   /* Event check fail addr and counter. */
   vex_state->host_EvC_FAILADDR = 0;
   vex_state->host_EvC_COUNTER  = 0;

   /* CPU Registers */
   vex_state->guest_R0   = 0; /* Constant zero */
   vex_state->guest_R1   = 0; /* Return address */
   vex_state->guest_R2   = 0; /* Thread pointer */
   vex_state->guest_R3   = 0; /* Stack pointer */
   vex_state->guest_R4   = 0; /* Argument registers / Return value */
   vex_state->guest_R5   = 0;
   vex_state->guest_R6   = 0; /* Argument registers */
   vex_state->guest_R7   = 0;
   vex_state->guest_R8   = 0;
   vex_state->guest_R9   = 0;
   vex_state->guest_R10  = 0;
   vex_state->guest_R11  = 0;
   vex_state->guest_R12  = 0; /* Temporary registers */
   vex_state->guest_R13  = 0;
   vex_state->guest_R14  = 0;
   vex_state->guest_R15  = 0;
   vex_state->guest_R16  = 0;
   vex_state->guest_R17  = 0;
   vex_state->guest_R18  = 0;
   vex_state->guest_R19  = 0;
   vex_state->guest_R20  = 0;
   vex_state->guest_R21  = 0; /* Reserved */
   vex_state->guest_R22  = 0; /* Frame pointer / Static register */
   vex_state->guest_R23  = 0; /* Static registers */
   vex_state->guest_R24  = 0;
   vex_state->guest_R25  = 0;
   vex_state->guest_R26  = 0;
   vex_state->guest_R27  = 0;
   vex_state->guest_R28  = 0;
   vex_state->guest_R29  = 0;
   vex_state->guest_R30  = 0;
   vex_state->guest_R31  = 0;

   vex_state->guest_PC   = 0; /* Program counter */

   /* FPU Registers */
   vex_state->guest_F0   = 0xffffffffffffffffUL; /* Argument registers / Return value */
   vex_state->guest_F1   = 0xffffffffffffffffUL;
   vex_state->guest_F2   = 0xffffffffffffffffUL; /* Argument registers */
   vex_state->guest_F3   = 0xffffffffffffffffUL;
   vex_state->guest_F4   = 0xffffffffffffffffUL;
   vex_state->guest_F5   = 0xffffffffffffffffUL;
   vex_state->guest_F6   = 0xffffffffffffffffUL;
   vex_state->guest_F7   = 0xffffffffffffffffUL;
   vex_state->guest_F8   = 0xffffffffffffffffUL; /* Temporary registers */
   vex_state->guest_F9   = 0xffffffffffffffffUL;
   vex_state->guest_F10  = 0xffffffffffffffffUL;
   vex_state->guest_F11  = 0xffffffffffffffffUL;
   vex_state->guest_F12  = 0xffffffffffffffffUL;
   vex_state->guest_F13  = 0xffffffffffffffffUL;
   vex_state->guest_F14  = 0xffffffffffffffffUL;
   vex_state->guest_F15  = 0xffffffffffffffffUL;
   vex_state->guest_F16  = 0xffffffffffffffffUL;
   vex_state->guest_F17  = 0xffffffffffffffffUL;
   vex_state->guest_F18  = 0xffffffffffffffffUL;
   vex_state->guest_F19  = 0xffffffffffffffffUL;
   vex_state->guest_F20  = 0xffffffffffffffffUL;
   vex_state->guest_F21  = 0xffffffffffffffffUL;
   vex_state->guest_F22  = 0xffffffffffffffffUL;
   vex_state->guest_F23  = 0xffffffffffffffffUL;
   vex_state->guest_F24  = 0xffffffffffffffffUL; /* Static registers */
   vex_state->guest_F25  = 0xffffffffffffffffUL;
   vex_state->guest_F26  = 0xffffffffffffffffUL;
   vex_state->guest_F27  = 0xffffffffffffffffUL;
   vex_state->guest_F28  = 0xffffffffffffffffUL;
   vex_state->guest_F29  = 0xffffffffffffffffUL;
   vex_state->guest_F30  = 0xffffffffffffffffUL;
   vex_state->guest_F31  = 0xffffffffffffffffUL;

   vex_state->guest_FCC0 = 0; /* Condition Flag Registers */
   vex_state->guest_FCC1 = 0;
   vex_state->guest_FCC2 = 0;
   vex_state->guest_FCC3 = 0;
   vex_state->guest_FCC4 = 0;
   vex_state->guest_FCC5 = 0;
   vex_state->guest_FCC6 = 0;
   vex_state->guest_FCC7 = 0;
   vex_state->guest_FCSR = 0; /* FP Control and Status Register */

   /* Various pseudo-regs mandated by Vex or Valgrind. */
   /* Emulation notes */
   vex_state->guest_EMNOTE = 0;
   vex_state->guest_COND   = 0;

   /* For clflush: record start and length of area to invalidate */
   vex_state->guest_CMSTART = 0;
   vex_state->guest_CMLEN   = 0;

   /* Used to record the unredirected guest address at the start of
      a translation whose start has been redirected.  By reading
      this pseudo-register shortly afterwards, the translation can
      find out what the corresponding no-redirection address was.
      Note, this is only set for wrap-style redirects, not for
      replace-style ones. */
   vex_state->guest_NRADDR = 0;
}


/*-----------------------------------------------------------*/
/*--- Describing the loongarch64 guest state, for the     ---*/
/*--- benefit of iropt and instrumenters                  ---*/
/*-----------------------------------------------------------*/

/* Figure out if any part of the guest state contained in minoff
   .. maxoff requires precise memory exceptions.  If in doubt return
   True (but this generates significantly slower code).

   We enforce precise exns for guest SP, PC and FP.

   Only SP is needed in mode VexRegUpdSpAtMemAccess.
*/

Bool guest_loongarch64_state_requires_precise_mem_exns ( Int minoff,
                                                         Int maxoff,
                                                         VexRegisterUpdates pxControl )
{
   Int sp_min = offsetof(VexGuestLOONGARCH64State, guest_R3);
   Int sp_max = sp_min + 8 - 1;
   if ( maxoff < sp_min || minoff > sp_max ) {
      /* no overlap with sp */
      if (pxControl == VexRegUpdSpAtMemAccess)
         return False;  /* We only need to check stack pointer. */
   } else {
      return True;
   }

   Int pc_min = offsetof(VexGuestLOONGARCH64State, guest_PC);
   Int pc_max = pc_min + 8 - 1;
   if ( maxoff < pc_min || minoff > pc_max ) {
      /* no overlap with pc */
   } else {
      return True;
   }

   Int fp_min = offsetof(VexGuestLOONGARCH64State, guest_R22);
   Int fp_max = fp_min + 8 - 1;
   if ( maxoff < fp_min || minoff > fp_max ) {
      /* no overlap with fp */
   } else {
      return True;
   }

   return False;
}

#define ALWAYSDEFD64(field)                            \
   { offsetof(VexGuestLOONGARCH64State, field),        \
      (sizeof ((VexGuestLOONGARCH64State*)0)->field) }

VexGuestLayout loongarch64Guest_layout = {
   /* Total size of the guest state, in bytes. */
   .total_sizeB = sizeof(VexGuestLOONGARCH64State),
   /* Describe the stack pointer. */
   .offset_SP = offsetof(VexGuestLOONGARCH64State, guest_R3),
   .sizeof_SP = 8,
   /* Describe the frame pointer. */
   .offset_FP = offsetof(VexGuestLOONGARCH64State, guest_R22),
   .sizeof_FP = 8,
   /* Describe the instruction pointer. */
   .offset_IP = offsetof(VexGuestLOONGARCH64State, guest_PC),
   .sizeof_IP = 8,
   /* Describe any sections to be regarded by Memcheck as
      'always-defined'. */
   .n_alwaysDefd = 6,
   /* ? :(  */
   .alwaysDefd = {
                  /* 0 */ ALWAYSDEFD64(guest_R0),
                  /* 1 */ ALWAYSDEFD64(guest_PC),
                  /* 2 */ ALWAYSDEFD64(guest_EMNOTE),
                  /* 3 */ ALWAYSDEFD64(guest_CMSTART),
                  /* 4 */ ALWAYSDEFD64(guest_CMLEN),
                  /* 5 */ ALWAYSDEFD64(guest_NRADDR),
                  }
};


/*-----------------------------------------------------------*/
/*--- loongarch64 guest helpers                           ---*/
/*-----------------------------------------------------------*/

/* Claim to be the following CPU, which is probably representative of
   the earliest loongarch64 offerings.

   CPU Family          : Loongson-64bit
   Model Name          : Loongson-3A5000LL
   CPU Revision        : 0x10
   FPU Revision        : 0x00
   CPU MHz             : 2300.00
   BogoMIPS            : 4600.00
   TLB Entries         : 2112
   Address Sizes       : 48 bits physical, 48 bits virtual
   ISA                 : loongarch32 loongarch64
   Features            : cpucfg lam ual fpu lsx lasx complex crypto lvz
   Hardware Watchpoint : yes, iwatch count: 8, dwatch count: 8
*/
ULong loongarch64_calculate_cpucfg ( ULong src )
{
   ULong res;
   switch (src) {
      case 0x0:
         res = 0x0014c010;
         break;
      case 0x1:
         res = 0x03f2f2fe;
         break;
      case 0x2:
         res = 0x007ccfc7;
         break;
      case 0x3:
         res = 0x0000fcff;
         break;
      case 0x4:
         res = 0x05f5e100;
         break;
      case 0x5:
         res = 0x00010001;
         break;
      case 0x6:
         res = 0x00007f33;
         break;
      case 0x10:
         res = 0x00002c3d;
         break;
      case 0x11:
         res = 0x06080003;
         break;
      case 0x12:
         res = 0x06080003;
         break;
      case 0x13:
         res = 0x0608000f;
         break;
      case 0x14:
         res = 0x060e000f;
         break;
      default:
         res = 0x00000000;
         break;
   }
   return (ULong)(Long)(Int)res;
}

static void swap_UChar ( UChar* a, UChar* b )
{
   UChar t = *a;
   *a = *b;
   *b = t;
}

ULong loongarch64_calculate_revb_2h ( ULong src )
{
   UChar* s = (UChar*)&src;
   swap_UChar(&s[0], &s[1]);
   swap_UChar(&s[2], &s[3]);
   return (ULong)(Long)(Int)src;
}

ULong loongarch64_calculate_revb_4h ( ULong src )
{
   UChar* s = (UChar*)&src;
   swap_UChar(&s[0], &s[1]);
   swap_UChar(&s[2], &s[3]);
   swap_UChar(&s[4], &s[5]);
   swap_UChar(&s[6], &s[7]);
   return src;
}

ULong loongarch64_calculate_revb_2w ( ULong src )
{
   UChar* s = (UChar*)&src;
   swap_UChar(&s[0], &s[3]);
   swap_UChar(&s[1], &s[2]);
   swap_UChar(&s[4], &s[7]);
   swap_UChar(&s[5], &s[6]);
   return src;
}

ULong loongarch64_calculate_revb_d ( ULong src )
{
   UChar* s = (UChar*)&src;
   swap_UChar(&s[0], &s[7]);
   swap_UChar(&s[1], &s[6]);
   swap_UChar(&s[2], &s[5]);
   swap_UChar(&s[3], &s[4]);
   return src;
}

static void swap_UShort ( UShort* a, UShort* b )
{
   UShort t = *a;
   *a = *b;
   *b = t;
}

ULong loongarch64_calculate_revh_2w ( ULong src )
{
   UShort* s = (UShort*)&src;
   swap_UShort(&s[0], &s[1]);
   swap_UShort(&s[2], &s[3]);
   return src;
}

ULong loongarch64_calculate_revh_d ( ULong src )
{
   UShort* s = (UShort*)&src;
   swap_UShort(&s[0], &s[3]);
   swap_UShort(&s[1], &s[2]);
   return src;
}

static ULong bitrev ( ULong src, ULong start, ULong end )
{
   int i, j;
   ULong res = 0;
   for (i = start, j = 1; i < end; i++, j++)
      res |= ((src >> i) & 1) << (end - j);
   return res;
}

ULong loongarch64_calculate_bitrev_4b ( ULong src )
{
   ULong res = bitrev(src, 0, 8);
   res |= bitrev(src, 8, 16);
   res |= bitrev(src, 16, 24);
   res |= bitrev(src, 24, 32);
   return (ULong)(Long)(Int)res;
}

ULong loongarch64_calculate_bitrev_8b ( ULong src )
{
   ULong res = bitrev(src, 0, 8);
   res |= bitrev(src, 8, 16);
   res |= bitrev(src, 16, 24);
   res |= bitrev(src, 24, 32);
   res |= bitrev(src, 32, 40);
   res |= bitrev(src, 40, 48);
   res |= bitrev(src, 48, 56);
   res |= bitrev(src, 56, 64);
   return res;
}

ULong loongarch64_calculate_bitrev_w ( ULong src )
{
   ULong res = bitrev(src, 0, 32);
   return (ULong)(Long)(Int)res;
}

ULong loongarch64_calculate_bitrev_d ( ULong src )
{
   return bitrev(src, 0, 64);
}


/*---------------------------------------------------------------*/
/*--- end                         guest_loongarch64_helpers.c ---*/
/*---------------------------------------------------------------*/
