/*
 *========================================================================
 * $Id: dieharder.c 279 2007-02-05 19:51:33Z rgb $
 *
 * See copyright in copyright.h and the accompanying file COPYING
 *========================================================================
 */

/*
 *========================================================================
 *  Project template.
 *========================================================================
 */

#include "dieharder.h"


/*
 * main() is just a shell for routines that parse the command line,
 * do all the requested work, and then exit cleanly.  main() is pretty
 * much all there is in cpu_rate.c
 */
int main(int argc, char *argv[]) 
{

 /*
  * Parse command line and set global variables
  */
 parsecl(argc,argv);

 /*
  * Note that most of my cpu_rates (except the terminally simple/stupid) 
  * have three phases after parsecl():
  *
  * Startup: Allocate memory, initialize all derivative variables from
  * command line values.  
  */
 startup();

 /*
  * Work: Do all the work.  In a complicated cpu_rate, project_work would
  * itself be a shell for a lot of other modular routines.
  */
 work();

 /* 
  * Quit:  Done. Clean up (if necessary) and exit.
 dieharder_quit();
  */

}
