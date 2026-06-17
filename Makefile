# LIFTOFF Makefile — convenience build targets
# Usage: make test | make bench | make ptx | make clean

NVCC     = nvcc
ARCH     = sm_75
FLAGS    = -O3 -arch=$(ARCH) --use_fast_math -std=c++17 -I./liftoff
DIAG     = -diag-suppress 177

.PHONY: all test bench ptx clean

all: test bench

test: liftoff_test
	./liftoff_test

bench: liftoff_bench
	./liftoff_bench

liftoff_test: liftoff/tests/correctness.cu
	$(NVCC) $(FLAGS) $(DIAG) $< -o $@

liftoff_bench: liftoff/bench/benchmark_main.cu
	$(NVCC) $(FLAGS) $(DIAG) $< -o $@

ptx:
	$(NVCC) $(FLAGS) -ptx liftoff/bench/benchmark_main.cu -o liftoff_bench.ptx
	@echo "=== Shuffle instructions ===" && grep -c "shfl" liftoff_bench.ptx
	@echo "=== Shared memory loads  ===" && grep -c "ld.shared" liftoff_bench.ptx
	@echo "=== Shared memory stores ===" && grep -c "st.shared" liftoff_bench.ptx

registers:
	$(NVCC) $(FLAGS) --ptxas-options=-v liftoff/bench/benchmark_main.cu -o /dev/null 2>&1 | grep "registers"

clean:
	rm -f liftoff_test liftoff_bench *.ptx *.csv *.png sweep
