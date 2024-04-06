histogram : histogram.cu
	nvcc.exe $<

run : a.exe
	./a.exe

clean :
	rm -f a.exe a.exp a.lib
