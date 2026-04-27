CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDFLAGS =

SEQ_SRC = sequential/jobshop_seq.c
PAR_SRC = parallel/jobshop_par.c
SEQ_BIN = sequential/jobshop_seq
PAR_BIN = parallel/jobshop_par

.PHONY: all clean test_seq test_par

all: $(SEQ_BIN) $(PAR_BIN)

$(SEQ_BIN): $(SEQ_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(PAR_BIN): $(PAR_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

test_seq: $(SEQ_BIN)
	@echo "=== Sequential: gg03 ==="
	./$(SEQ_BIN) data/gg03.jss data/gg03_seq.output
	@cat data/gg03_seq.output
	@echo ""
	@echo "=== Sequential: ft06 ==="
	./$(SEQ_BIN) data/ft06.jss data/ft06_seq.output
	@cat data/ft06_seq.output

test_par: $(PAR_BIN)
	@echo "=== Parallel (4 threads): gg03 ==="
	./$(PAR_BIN) data/gg03.jss data/gg03_par.output 4
	@cat data/gg03_par.output
	@echo ""
	@echo "=== Parallel (4 threads): ft06 ==="
	./$(PAR_BIN) data/ft06.jss data/ft06_par.output 4
	@cat data/ft06_par.output

clean:
	rm -f $(SEQ_BIN) $(PAR_BIN) data/*.output
