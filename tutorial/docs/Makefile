# Build the RecoCrysp tutorial documents. Run pdflatex twice so the table of
# contents and cross-references resolve.
DOCS = tutorial_basis tutorial_example_attenuation tutorial_example_resolution

all: $(addsuffix .pdf,$(DOCS))

%.pdf: %.tex $(wildcard tbsrc/*.tex) recocrysp_tutorial.sty
	pdflatex -interaction=nonstopmode -halt-on-error $*.tex
	pdflatex -interaction=nonstopmode -halt-on-error $*.tex

clean:
	rm -f *.aux *.log *.out *.toc tbsrc/*.aux

distclean: clean
	rm -f $(addsuffix .pdf,$(DOCS))

.PHONY: all clean distclean
