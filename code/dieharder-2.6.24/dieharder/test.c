/*
 *========================================================================
 * $Id: help.c 181 2006-07-12 16:41:25Z rgb $
 *
 * See copyright in copyright.h and the accompanying file COPYING
 *========================================================================
 */

#include "dieharder.h"

/*
 * Write a standard test header.
 */
void show_test_header(Dtest *dtest,Test **test)
{

 /*
  * Show standard test description
  */
 printf("%s",dtest->description);

 /*
  * This is the non-custom part of the header, which should be enough
  * for nearly all tests by design.
  */
 printf("#                        Run Details\n");

 /*
  * Show the generator being tested, handling file_input generators
  * separately.
  */
 if(strncmp("file_input",gsl_rng_name(rng),10) == 0){
   printf("# Random number generator tested: %s\n",gsl_rng_name(rng));
   printf("# File %s contains %u rands of %c type.\n",filename,filecount,filetype);
 } else {
   printf("# Random number generator tested: %s\n",gsl_rng_name(rng));
 }

 /*
  * Now show both current and default values for tsamples and psamples.
  */
 printf("# Samples per test pvalue = %u (test default is %u)\n",test[0]->tsamples,dtest->tsamples_std);
 printf("# P-values in final KS test = %u (test default is %u)\n",test[0]->psamples,dtest->psamples_std);

}


/*
 * Write a standard test footer.  Arguments include the test
 * variables, the test pvalue, the ks_pvalues, and a test description.
 */
void show_test_results(Dtest *dtest,Test **test)
{

 int i,j,k;

 /*
  * There may be more than one statistic (final p-value) generated by
  * this test; we loop over all of them.
  */
 /* printf("In show_test_results for %s for %u statistics\n",dtest->name,dtest->nkps); */
 for(i=0;i<dtest->nkps;i++){

   /*
    * Display histogram of ks p-values (optional)
    */
   if(hist_flag){
     /*
     for(j=0;j<test[i]->psamples;j++){
       printf("pvalue = %f\n",test[i]->pvalues[j]);
     }*/
     histogram(test[i]->pvalues,test[i]->psamples,0.0,1.0,10,"p-values");
   }

   if(strncmp("file_input",gsl_rng_name(rng),10) == 0){
     printf("# %u rands were used in this test\n",file_input_get_rtot(rng));
     printf("# The file %s was rewound %u times\n",gsl_rng_name(rng),file_input_get_rewind_cnt(rng));
   }

   printf("#                          Results\n");
   if(test[i]->psamples == 1){
     printf("Single test: p = %8.6f\n",test[i]->ks_pvalue);
     printf("Assessment: ");
     /* Work through some ranges here */
     if(test[i]->ks_pvalue < 0.0001 || test[i]->ks_pvalue > 0.9999){
       printf("FAILED at < 0.02%% for %s\n",dtest->name);
     } else if(test[i]->ks_pvalue < 0.01 || test[i] -> ks_pvalue > 0.99){
       printf("POOR at < 2%% for %s\n",dtest->name);
       printf("Recommendation:  Repeat test to verify failure.\n");
     } else {
       printf("PASSED at > 2%% for %s\n",dtest->name);
     }
   } else {
     printf("Kuiper KS: p = %8.6f\n",test[i]->ks_pvalue);
     printf("Assessment: ");
     /* Work through some ranges here */
     if(test[i]->ks_pvalue < 0.0001){
       printf("FAILED at < 0.01%% for %s\n",dtest->name);
     } else if(test[i]->ks_pvalue < 0.01){
       printf("POOR at < 1%% for %s\n",dtest->name);
       printf("Recommendation:  Repeat test to verify failure.\n");
     } else if(test[i]->ks_pvalue < 0.05){
       printf("POSSIBLY WEAK at < 5%% for %s\n",dtest->name);
       printf("Recommendation:  Repeat test to verify failure.\n");
     } else {
       printf("PASSED at > 5%% for %s\n",dtest->name);
     }
   }
 }

}

/*
 * Cruft, but cannoot be removed until we are done
 */
void test_footer(Dtest *dtest,double pvalue,double *pvalues)
{

 /*
  * Display histogram of ks p-values (optional)
  */
 if(hist_flag){
   histogram(pvalues,psamples,0.0,1.0,10,"p-values");
 }

 if(strncmp("file_input",gsl_rng_name(rng),10) == 0){
   printf("# %u rands were used in this test\n",file_input_get_rtot(rng));
   printf("# The file %s was rewound %u times\n",gsl_rng_name(rng),file_input_get_rewind_cnt(rng));
 }

 printf("#                          Results\n");
 printf("# Kuiper KS: p = %8.6f for %s\n",pvalue,dtest->name);
 printf("# Assessment:\n");
 /* Work through some ranges here */
 if(pvalue < 0.0001){
   printf("# FAILED at < 0.01%%.\n");
 } else if(pvalue < 0.01){
   printf("# POOR at < 1%%.\n");
   printf("# Recommendation:  Repeat test to verify failure.\n");
 } else if(pvalue < 0.05){
   printf("# POSSIBLY WEAK at < 5%%.\n");
   printf("# Recommendation:  Repeat test to verify failure.\n");
 } else {
   printf("# PASSED at > 5%%.\n");
 }

}
/*
 * Write a standard test header.
 */
void test_header(Dtest *dtest)
{

 /*
  * Show standard test description
  */
 printf("%s",dtest->description);

 /*
  * This is the non-custom part of the header, which should be enough
  * for nearly all tests by design.
  */
 printf("#                        Run Details\n");

 /*
  * Show the generator being tested, handling file_input generators
  * separately.
  */
 if(strncmp("file_input",gsl_rng_name(rng),10) == 0){
   printf("# Random number generator tested: %s\n",gsl_rng_name(rng));
   printf("# File %s contains %u rands of %c type.\n",filename,filecount,filetype);
 } else {
   printf("# Random number generator tested: %s\n",gsl_rng_name(rng));
 }

 /*
  * Now show both current and default values for tsamples and psamples.
  */
 printf("# Samples per test pvalue = %u (test default is %u)\n",tsamples,dtest->tsamples_std);
 printf("# P-values in final KS test = %u (test default is %u)\n",psamples,dtest->psamples_std);

}

