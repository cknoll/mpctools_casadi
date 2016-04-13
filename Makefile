# Makefile for the zip distribution.

# List source files here.
MPCTOOLS_SRC := $(addprefix mpctools/, __init__.py colloc.py plots.py \
                  solvers.py tools.py util.py)

EXAMPLES := airplane.py ballmaze.py cstr.py cstr_startup.py cstr_nmpc_nmhe.py \
            collocationexample.py comparison_casadi.py comparison_mtc.py \
            econmpc.py example2-8.py mheexample.py mpcexampleclosedloop.py \
            mpcmodelcomparison.py nmheexample.py nmpcexample.py \
            periodicmpcexample.py vdposcillator.py runall.py

DOC_TEX := $(addprefix doc/, install.tex cheatsheet.tex introslides.tex \
             octave-vs-python.tex)
DOC_PDF := $(DOC_TEX:.tex=.pdf)

INSTALLER_FILES := $(addprefix installer/, casadiinstaller.py casadisetup.py \
                     README.txt)

CSTR_MATLAB_FILES := $(addprefix cstr-matlab/, main.m massenbal.m \
                       massenbalstst.m partial.m)

MISC_FILES := COPYING.txt mpctoolssetup.py cstr.m README.md

MPC_TOOLS_CASADI_FILES := $(MPCTOOLS_SRC) $(EXAMPLES) $(DOC_PDF) \
                          $(INSTALLER_FILES) $(CSTR_MATLAB_FILES) $(MISC_FILES)

ZIPNAME := mpc-tools-casadi.zip

# Define zip rule.
$(ZIPNAME) : $(MPC_TOOLS_CASADI_FILES)
	@echo "Building zip distribution."
	@python makezip.py --name $(ZIPNAME) $(MPC_TOOLS_CASADI_FILES) || rm -f $(ZIPNAME)

UPLOAD_COMMAND := POST https://api.bitbucket.org/2.0/repositories/rawlings-group/mpc-tools-casadi/downloads
upload : $(ZIPNAME) bitbucketuser
	curl -v -u $(shell cat bitbucketuser) -X $(UPLOAD_COMMAND) -F files=@"$(ZIPNAME)"
.PHONY : upload

# Rules for documentation pdfs.
$(DOC_PDF) : %.pdf : %.tex
	@echo "Making $@."
	@python latex2pdf.py --display errors --dir $(@D) $<

# Rule to make Matlab versions of Octave CSTR example.
$(CSTR_MATLAB_FILES) : cstr.m
	@echo "Making Matlab CSTR files."
	@cd doc && python matlabify.py

# Documentation dependencies.
doc/introslides.pdf : cstr_octave.pdf cstr_python.pdf vdposcillator_lmpc.pdf \
                      vdposcillator_nmpc.pdf cstr_startup.pdf
doc/cheatsheet.pdf : doc/sidebyside.tex
doc/octave-vs-python.pdf : doc/sidebyside-cstr.tex

# Define rules for intermediate files.
cstr_octave.pdf : cstr.m
	@echo "Making $@."
	@octave $<

cstr_python.pdf : cstr.py
	@echo "Making $@."
	@python $< --ioff

vdposcillator_lmpc.pdf vdposcillator_nmpc.pdf : vdposcillator.py
	@echo "Making vdposcillator pdfs."
	@python $< --ioff

cstr_startup.pdf : cstr_startup.py
	@echo "Making $@."
	@python $< --ioff

doc/sidebyside.tex : comparison_casadi.py comparison_mtc.py
	@echo "Making $@."
	@cd doc && python doSourceComparison.py $(@F)

doc/sidebyside-cstr.tex : cstr.m cstr.py
	@echo "Making $@."
	@cd doc && python doSourceComparison.py $(@F)

# Define cleanup rules.
TEXSUFFIXES := .log .aux .toc .vrb .synctex.gz .snm .nav .out
TEX_MISC := $(foreach doc, $(basename $(DOC_TEX)), $(addprefix $(doc), $(TEXSUFFIXES)))
OTHER_MISC := doc/sidebyside.tex doc/sidebyside-cstr.tex
clean :
	@rm -f $(ZIPNAME) $(TEX_MISC) $(OTHER_MISC)
.PHONY : clean

PDF_MISC := $(DOC_PDF) cstr_octave.pdf cstr_python.pdf vdposcillator_lmpc.pdf \
            vdposcillator_nmpc.pdf cstr_startup.pdf
realclean : clean
	@rm -f $(PDF_MISC) $(CSTR_MATLAB_FILES)
.PHONY : realclean

