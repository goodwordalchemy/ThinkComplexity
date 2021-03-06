/*
 *========================================================================
 * $Id: add_my_types.c 279 2007-02-05 19:51:33Z rgb $
 *
 * See copyright in copyright.h and the accompanying file COPYING
 *========================================================================
 */

#include "dieharder.h"

/*
 * This routine just adds new RNG's onto the GSL types list.
 * Apparently this is all that is needed -- they subsequently
 * just "work" in the gsl call format.  Note that there DOES
 * have to be a subroutine set with its own state and so forth just
 * like the one in dev_random.c (for the linus entropy generator).
 *
 * From gsl types.c -- N apparently a hard-coded value.  This means
 * that we cannot CURRENTLY go over 100 generators without e.g.
 * increasing N in the gsl sources.  They do leave some room for
 * new ones, but not a lot.  Fortunately, I probably won't ever
 * fill N myself personally...
 */
#define N 100

/* #define GSL_VAR */
/* List new rng types to be added. */
GSL_VAR const gsl_rng_type *gsl_rng_dev_random;
GSL_VAR const gsl_rng_type *gsl_rng_dev_urandom;
GSL_VAR const gsl_rng_type *gsl_rng_empty_random;
GSL_VAR const gsl_rng_type *gsl_rng_file_input;
GSL_VAR const gsl_rng_type *gsl_rng_file_input_raw;

void add_my_types()
{

 int i = 0;

 /*
  * Get to the end of types
  */
 while(types[i] != NULL){
   i++;
 }
 /*
  * and add the new ones...
  */
 types[i] = (gsl_rng_dev_random);
 if(verbose) printf("# add_my_types():  Added type %d = %s\n",i,types[i]->name);
 i++;
 types[i] = (gsl_rng_dev_urandom);
 if(verbose) printf("# add_my_types():  Added type %d = %s\n",i,types[i]->name);
 i++;
 types[i] = (gsl_rng_empty_random);
 if(verbose) printf("# add_my_types():  Added type %d = %s\n",i,types[i]->name);
 i++;
 types[i] = (gsl_rng_file_input);
 if(verbose) printf("# add_my_types():  Added type %d = %s\n",i,types[i]->name);
 i++;
 types[i] = (gsl_rng_file_input_raw);
 if(verbose) printf("# add_my_types():  Added type %d = %s\n",i,types[i]->name);
 i++;
 
}
