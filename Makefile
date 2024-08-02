all: small_haar generate_image

small_haar:
	gcc small_haar.c -o small_haar

generate_image: haar_generator
	./haar_generator

haar_generator:
	gcc haarGenerator.c -o haar_generator

clean:
	rm -rf *.o; rm small_haar; rm haar_generator; rm *.in