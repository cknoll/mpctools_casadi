# Makes all the documentation.

PDFLATEX := pdflatex --interaction=nonstopmode

# Pdfs.
tex := cheatsheet.tex install.tex introslides.tex octave-vs-python.tex
pdf := $(tex:.tex=.pdf)

all : $(pdf)
.PHONY : all

$(pdf) : %.pdf : %.tex
	@echo Making $@
	@$(PDFLATEX) $< > /dev/null
	@$(PDFLATEX) $< > /dev/null
	@$(PDFLATEX) $<
        # Yes, we could be smarter than just running three times.

# Included .tex files.
sidebyside := sidebyside.tex sidebyside-cstr.tex
$(sidebyside) : %.tex : doSourceComparison.py
	@echo Making $@
	@python $< $@

# Included .pdf files.
vdp := ../vdposcillator_lmpc.pdf ../vdposcillator_nmpc.pdf
$(vdp) : ../vdposcillator.py
	@echo Making $@
	@cd .. && python vdposcillator.py --ioff

../cstr_python.pdf : ../cstr.py
	@echo Making $@
	@cd .. && python cstr.py --ioff

../cstr_octave.pdf : ../cstr.m
	@echo Making $@
	@cd .. && octave cstr.m

../cstr_startup.pdf : ../cstr_startup.py
	@echo Making $@
	@cd .. && python cstr_startup.py --ioff

# Extra dependencies.
introslides_figures = ../vdposcillator_nmpc.pdf ../vdposcillator_lmpc.pdf \
    ../cstr_python.pdf ../cstr_octave.pdf ../cstr_startup.pdf
introslides.pdf : $(introslides_figures)

cheatsheet.pdf : sidebyside.tex

octave-vs-python.pdf : sidebyside-cstr.tex

# Cleanup.
texbase := $(basename $(tex))
texsuffixes := aux log nav out snm synctex.gz vrb toc
texfiles := $(foreach f, $(texbase), $(addprefix $(f)., $(texsuffixes)))
clean :
	rm -f $(texfiles)
.PHONY : clean

realclean : clean
	rm -f $(introslides_figures)
	rm -f $(sidebyside)
.PHONY : clean

allclean : realclean
	rm -f $(pdf)
.PHONY : allclean

# cstr-matlab rules.
cstr-matlab : matlabify.py ../cstr.m
	@echo Making cstr-matlab
	@python matlabify.py
.PHONY : cstr-matlab


