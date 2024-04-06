histogram : histogram.cu
	nvcc $<

run : a.exe
	./a.out

clean :
	rm -f a.out a.exp a.lib
