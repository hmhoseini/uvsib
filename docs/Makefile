# Build PDF documentation from the .md files in this directory.
#
# Targets:
#   make            build all *.md -> *.pdf (default)
#   make watch      rebuild on file changes (needs `entr`)
#   make clean      remove generated PDFs
#
# Override variables on the command line, e.g.
#   make MAINFONT='TeX Gyre Heros' MONOFONT='TeX Gyre Cursor'

PANDOC   ?= pandoc
ENGINE   ?= xelatex
MAINFONT ?= DejaVu Sans
MONOFONT ?= DejaVu Sans Mono

MD  := $(wildcard *.md)
PDF := $(MD:.md=.pdf)

PANDOC_FLAGS := --pdf-engine=$(ENGINE) \
                --toc \
                --number-sections \
                -V geometry:margin=2cm \
                -V mainfont="$(MAINFONT)" \
                -V monofont="$(MONOFONT)" \
                -V colorlinks=true

.PHONY: all watch clean
all: $(PDF)

%.pdf: %.md
	$(PANDOC) $(PANDOC_FLAGS) -o $@ $<

watch:
	@command -v entr >/dev/null || { echo "install 'entr' first (apt install entr)"; exit 1; }
	ls $(MD) | entr -c $(MAKE)

clean:
	rm -f $(PDF)
