NOTE = joseph3d_note

all: $(NOTE).pdf

$(NOTE).pdf: $(NOTE).tex
	pdflatex -interaction=nonstopmode $(NOTE).tex
	pdflatex -interaction=nonstopmode $(NOTE).tex

clean:
	rm -f $(NOTE).aux $(NOTE).log $(NOTE).out $(NOTE).toc

distclean: clean
	rm -f $(NOTE).pdf

.PHONY: all clean distclean
