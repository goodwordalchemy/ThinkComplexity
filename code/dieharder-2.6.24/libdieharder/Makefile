#========================================================================
# This is a Makefile for the libdieharder library, part of the
# overall dieharder package.  It has a very abbreviated set of targets.
#
# make          alone should build the application.
#
# make clean    deletes the application and all object files
# make install  strips and installs application and a man page
# make printout prints out all source and include files
# 
# These two commands execute in the toplevel project directory only.
# make svn      does a svn commit and creates the timestamp $(SVN)
# make sync     does a svn commit and rsyncs to list of hosts
#========================================================================
PROJECT = dieharder
DIR = libdieharder

#========================================================================
# This is essential.  The rpmbuild overrides it, but we have to make
# build "work" when executed only in the source tree directory itself.
# This isn't easy, since the dependences are more than a bit scattered.
# We therefore point to the directory one level up, where we should
# find a ./lib, ./include, ./share and ./bin directory tree from which
# the various dieharder files will actually be assembled into an rpm
# with PREFIX=/usr (for example).
#========================================================================
BUILDROOT=../buildroot
PREFIX=$(BUILDROOT)/usr
LIBDIR=$(PREFIX)/lib
INCDIR=$(PREFIX)/include
INCTIME=include.time

SVNTREE = $(HOME)/Src/svn-tree
SVNPATH = $(SVNTREE)/$(PROJECT)
SVNTIME = $(DIR:=.svn.time)

# This is the library from which both wulfstat and wulflogger (and maybe
# other stuff in the future) is built.
PROGMAN = libdieharder.3
# (FC "requires" no static library build)
# PROGLIB_A = libdieharder.a
PROGLIB_SO = libdieharder.so
PROGLIB_SONAME = libdieharder.so.$(VERSION_MAJOR)
PROGLIB_SO_VERSION = libdieharder.so.$(VERSION_MAJOR).$(VERSION_MINOR)

# This is automagically set in the toplevel build.  Do not edit by
# hand.
VERSION_MAJOR=2
VERSION_MINOR=6.24
RELEASE=1

#========================================================================
# Define all sources.  Note that we choose to depend on ALL the includes
# in the include directory, which cannot be empty or this will barf.
#========================================================================
LIBSOURCES = $(shell ls *.c  2>&1 | sed -e "/\/bin\/ls:/d")
LIBOBJECTS = $(LIBSOURCES:.c=.o)

DEFINES = -DVERSION_MAJOR=$(VERSION_MAJOR) -DVERSION_MINOR=$(VERSION_MINOR) \
          -DRELEASE=$(RELEASE)

#========================================================================
# Define parameters and directives needed in compile/link steps.
#========================================================================
# C Compiler
CC = gcc

# Compile flags (use fairly standard -O3 as default)
CFLAGS = -O3 $(DEFINES) -I $(INCDIR)

# Linker flags
LDFLAGS = -g -fpic -shared -Wl,-soname,$(PROGLIB_SONAME)

# Libraries
LIBS = -lgsl -lgslcblas -lm

#========================================================================
# List of variants one can make.  all is the default.  We always
# presume the simplest of dependencies and remake if includes change
# for example.
#========================================================================
all: $(INCTIME) $(PROGLIB_SO_VERSION) # $(PROGLIB_A)

# This has to ALWAYS run before a build.
$(INCTIME):
	(install -d $(INCDIR)/dieharder; \
	install -m 644 ../include/dieharder/*.h $(INCDIR)/dieharder;)

# FC "requires" no static lib build/install	
# $(PROGLIB_A): $(INCTIME) $(LIBOBJECTS) $(LIBINCLUDES)
# 	ar r $(PROGLIB_A) $(LIBOBJECTS)
# 	ranlib $(PROGLIB_A)

$(PROGLIB_SO_VERSION): $(INCTIME) $(LIBOBJECTS) $(LIBINCLUDES)
	gcc $(LDFLAGS) -o $(PROGLIB_SO_VERSION) $(LIBOBJECTS)

#========================================================================
# The only safe place to do commits is in the toplevel directory
#========================================================================
svn:
	(cd ..;make svn)

sync:
	(cd ..;make sync)

#========================================================================
# printout makes an enscript -2r printout of SOURCES and
# and INCLUDES.  Use lpr if you don't have enscript
#========================================================================
LPR = enscript -2r
# LPR = lpr
printout:
	$(LPR) $(LIBSOURCES) $(LIBINCLUDES)

#========================================================================
#  A standard cleanup target
#========================================================================
clean:
	rm -f core $(PROGLIB) $(PROGLIB_NAME) $(LIBOBJECTS) $(PROGMAN).gz

#========================================================================
# This is critical.  For the toplevel rpm build to succeed,
#  make PREFIX=/usr install
# (run by rpmbuild from the specfile) has to work, in the right order.
# This target has to install precisely the files required by the
# specfile for the dieharder package, in precisely the right locations.
#========================================================================
install: $(PROGLIB_SO_VERSION) $(PROGLIB_A) $(PROGMAN)
	(install -d $(LIBDIR); \
	install -m 755 $(PROGLIB_SO_VERSION) $(LIBDIR)/$(PROGLIB_SO_VERSION); \
	install -d $(INCDIR)/dieharder; \
	install -m 644 ../include/dieharder/*.h $(INCDIR)/dieharder; \
	install -d $(PREFIX)/share/man/man3; \
	gzip -c $(PROGMAN) > $(PROGMAN).gz; \
	install -m 644 $(DIR).3.gz $(PREFIX)/share/man/man3)

# FC requires no static lib install
# 	install -m 755 $(PROGLIB_A) $(LIBDIR)/$(PROGLIB_A); \
#========================================================================
# We give all generic rules below.  Currently we only need a rule for 
# objects.
#========================================================================
%.o:%.c $(LIBINCLUDES) Makefile
	$(CC) -fpic -c $(CFLAGS) $<
