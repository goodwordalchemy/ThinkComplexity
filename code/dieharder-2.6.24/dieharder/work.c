/*
 *========================================================================
 * $Id: work.c 279 2007-02-05 19:51:33Z rgb $
 *
 * See copyright in copyright.h and the accompanying file COPYING
 *========================================================================
 */

/*
 *========================================================================
 * This should be a nice, big case switch where we add EACH test
 * we might want to do and either just configure and do it or
 * prompt for input (if absolutely necessary) and then do it.
 *========================================================================
 */

#include "dieharder.h"

void work()
{

/*
 if(output == YES){
   output_rnds();
 }
 */

 if(all == YES){
   run_rgb_timing();
   run_rgb_persist();
   run_rgb_bitdist();
   run_diehard_birthdays();
   run_diehard_operm5();
   run_diehard_rank_32x32();
   run_diehard_rank_6x8();
   run_diehard_bitstream();
   run_diehard_opso();
   run_diehard_oqso();
   run_diehard_dna();
   run_diehard_count_1s_stream();
   run_diehard_count_1s_byte();
   run_diehard_parking_lot();
   run_diehard_2dsphere();
   run_diehard_3dsphere();
   run_diehard_squeeze();
   run_diehard_sums();
   run_diehard_runs();
   run_diehard_craps();
   run_marsaglia_tsang_gcd();
   /* marsaglia_tsang_gorilla(); */
   run_sts_monobit();
   run_sts_runs();
   run_user_template();
   exit(0);
 }

 switch(diehard){
   case DIEHARD_BDAY:
     run_diehard_birthdays();
     exit(0);
     break;
   case DIEHARD_OPERM5:
     run_diehard_operm5();
     exit(0);
     break;
   case DIEHARD_RANK_32x32:
     run_diehard_rank_32x32();
     exit(0);
     break;
   case DIEHARD_RANK_6x8:
     run_diehard_rank_6x8();
     exit(0);
     break;
   case DIEHARD_BITSTREAM:
     run_diehard_bitstream();
     exit(0);
     break;
   case DIEHARD_OPSO:
     run_diehard_opso();
     exit(0);
     break;
   case DIEHARD_OQSO:
     run_diehard_oqso();
     exit(0);
     break;
   case DIEHARD_DNA:
     run_diehard_dna();
     exit(0);
     break;
   case DIEHARD_COUNT_1S_STREAM:
     run_diehard_count_1s_stream();
     exit(0);
     break;
   case DIEHARD_COUNT_1S_BYTE:
     run_diehard_count_1s_byte();
     exit(0);
     break;
   case DIEHARD_PARKING_LOT:
     run_diehard_parking_lot();
     exit(0);
     break;
   case DIEHARD_2DSPHERE:
     run_diehard_2dsphere();
     exit(0);
     break;
   case DIEHARD_3DSPHERE:
     run_diehard_3dsphere();
     exit(0);
     break;
   case DIEHARD_SQUEEZE:
     run_diehard_squeeze();
     exit(0);
     break;
   case DIEHARD_SUMS:
     run_diehard_sums();
     exit(0);
     break;
   case DIEHARD_RUNS:
     run_diehard_runs();
     exit(0);
     break;
   case DIEHARD_CRAPS:
     run_diehard_craps();
     exit(0);
     break;
   case MARSAGLIA_TSANG_GCD:
     run_marsaglia_tsang_gcd();
     exit(0);
     break;
   case MARSAGLIA_TSANG_GORILLA:
     /* marsaglia_tsang_gorilla(); */
     exit(0);
     break;
   default:
     break;
 }

 switch(rgb){
   case RGB_TIMING:
     run_rgb_timing();
     exit(0);
     break;
   case RGB_PERSIST:
     run_rgb_persist();
     exit(0);
     break;
   case RGB_BITDIST:
     run_rgb_bitdist();
     exit(0);
     break;
/*   case RGB_LMN:
     rgb_lmn();
     exit(0);
     break; */
   default:
     break;
 }

 switch(sts){
   case STS_MONOBIT:
     run_sts_monobit();
     exit(0);
     break;
   case STS_RUNS:
     run_sts_runs();
     exit(0);
     break;
   default:
     break;
 }

 switch(user){
   case USER_TEMPLATE:
     run_user_template();
     exit(0);
     break;
   default:
     break;
 }

 list_rngs();

}
