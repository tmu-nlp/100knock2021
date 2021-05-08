paste -d "\t" result/col1.txt  result/col2.txt > result/knock13_check.txt
diff -s result/knock13.txt result/knock13_check.txt