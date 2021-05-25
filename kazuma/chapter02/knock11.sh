sed "s/\t/ /g" data/popular-names.txt > result/knock11_check.txt
# expand data/popular-names.txt > result/knock11_check_expand.txt
diff -s result/knock11_check.txt result/knock11.txt