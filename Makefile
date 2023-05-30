all: rpt clean

rpt:
	pdflatex -interaction batchmode rapport.tex
	pdflatex -interaction batchmode rapport.tex

verbose:
	pdflatex rapport.tex
	pdflatex rapport.tex

test:
	@printf "==== TESTS PART 1 ====\n"
	@python3 src/test1.py
	@printf "\n==== TESTS PART 2a ====\n"
	@python3 src/test2a.py
	@printf "\n==== TESTS PART 2c ====\n"
	@python3 src/test2c.py

clean:
	rm -rf *.log *.aux *.toc
	rm -f chapters/*.aux